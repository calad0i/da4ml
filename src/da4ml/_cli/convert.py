import argparse
import json
from pathlib import Path

import numpy as np


def to_da4ml(
    model_path: Path,
    path: Path,
    n_test_sample: int,
    period: float,
    unc: float,
    flavor: str,
    latency_cutoff: int,
    part_name: str,
    verbose: int = 1,
    rtl_validation: bool = False,
    hwconf: tuple[int, int, int] = (1, -1, -1),
    hard_dc: int = 2,
    openmp: bool = True,
    n_threads: int = 4,
    metadata=None,
    inputs_kif: tuple[int, int, int] | None = None,
):
    from da4ml.cmvm.types import CombLogic
    from da4ml.codegen import RTLModel
    from da4ml.converter import trace_model
    from da4ml.trace import HWConfig, comb_trace

    if model_path.suffix in {'.h5', '.keras'}:
        import hgq  # noqa: F401
        import keras

        model: keras.Model = keras.models.load_model(model_path, compile=False)  # type: ignore
        if verbose > 1:
            model.summary()
        inp, out = trace_model(model, HWConfig(*hwconf), {'hard_dc': hard_dc}, verbose > 1, inputs_kif=inputs_kif)
        comb = comb_trace(inp, out)

    elif model_path.suffix == '.json':
        comb = CombLogic.load(model_path)
        model = None  # type: ignore

    else:
        raise ValueError(f'Unsupported model file format: {model_path}')

    rtl_model = RTLModel(
        comb,
        'model',
        path,
        flavor=flavor,
        latency_cutoff=latency_cutoff,
        print_latency=True,
        clock_uncertainty=unc / 100,
        clock_period=period,
        part_name=part_name,
    )
    rtl_model.write(metadata)
    if verbose > 1:
        print(rtl_model)
        print('Model written')
    if not n_test_sample:
        return

    if model is not None:
        data_in = [np.random.rand(n_test_sample, *inp.shape[1:]).astype(np.float32) * 64 - 32 for inp in model.inputs]
        if len(data_in) == 1:
            data_in = data_in[0]
        y_keras = model.predict(data_in, batch_size=16384, verbose=0)  # type: ignore

        if isinstance(y_keras, list):
            y_keras = np.concatenate([y.reshape(n_test_sample, -1) for y in y_keras], axis=1)
        else:
            y_keras = y_keras.reshape(n_test_sample, -1)
        y_comb = comb.predict(data_in, n_threads=n_threads)

        total = y_comb.size
        mask = y_comb != y_keras
        ndiff = np.sum(mask)
        if ndiff:
            n_nonzero = np.sum(y_keras != 0)
            abs_diff = np.abs(y_comb - y_keras)[mask]
            rel_diff = abs_diff / (np.abs(y_keras[np.where(mask)]) + 1e-6)

            max_diff, max_rel_diff = np.max(abs_diff), np.max(rel_diff)
            mean_diff, mean_rel_diff = np.mean(abs_diff), np.mean(rel_diff)
            print(
                f'[WARNING] {ndiff}/{total} ({n_nonzero}) mismatches ({max_diff=}, {max_rel_diff=}, {mean_diff=}, {mean_rel_diff=})'
            )
        else:
            max_diff = max_rel_diff = mean_diff = mean_rel_diff = 0.0
            if verbose:
                print(f'[INFO] DAIS simulation passed: [0/{total}] mismatches.')
        with open(path / 'mismatches.json', 'w') as f:
            json.dump(
                {
                    'n_total': int(total),
                    'n_mismatch': int(ndiff),
                    'max_diff': float(max_diff),
                    'max_rel_diff': float(max_rel_diff),
                    'mean_diff': float(mean_diff),
                    'mean_rel_diff': float(mean_rel_diff),
                },
                f,
            )
    else:
        if not rtl_validation:
            return
        data_in = np.random.rand(n_test_sample, comb.shape[0]).astype(np.float32) * 64 - 32
        y_comb = comb.predict(data_in, n_threads=n_threads)
        total = y_comb.size

    if not rtl_validation:
        return

    if verbose > 1:
        print('Verilating...')
    for _ in range(3):
        try:
            rtl_model._compile(nproc=n_threads, openmp=openmp)
            break
        except RuntimeError:
            pass
    y_da4ml = rtl_model.predict(data_in)
    if not np.all(y_comb == y_da4ml):
        raise RuntimeError(f'[CRITICAL ERROR] RTL validation failed: {np.sum(y_comb != y_da4ml)}/{total} mismatches!')
    if verbose:
        print(f'[INFO]  RTL validation passed: [0/{total}] mismatches.')


def convert_main(args):
    args.outdir.mkdir(parents=True, exist_ok=True)
    hw_conf = tuple(args.hw_config)
    if args.metadata is not None:
        with open(args.metadata) as f:
            metadata = json.load(f)
    else:
        metadata = None

    to_da4ml(
        args.model,
        args.outdir,
        args.n_test_sample,
        args.clock_period,
        args.clock_uncertainty,
        latency_cutoff=args.latency_cutoff,
        part_name=args.part_name,
        flavor=args.flavor,
        verbose=args.verbose,
        rtl_validation=args.validate_rtl,
        hwconf=hw_conf,
        hard_dc=args.delay_constraint,
        openmp=not args.no_openmp,
        n_threads=args.n_threads,
        metadata=metadata,
        inputs_kif=args.inputs_kif,
    )


def _add_convert_args(parser: argparse.ArgumentParser):
    parser.add_argument('model', type=Path, help='Path to the Keras model file (.h5 or .keras)')
    parser.add_argument('outdir', type=Path, help='Output directory')
    parser.add_argument('--n-test-sample', '-n', type=int, default=131072, help='Number of test samples for validation')
    parser.add_argument('--clock-period', '-c', type=float, default=5.0, help='Clock period in ns')
    parser.add_argument('--clock-uncertainty', '-unc', type=float, default=10.0, help='Clock uncertainty in percent')
    parser.add_argument('--flavor', type=str, default='verilog', help='Flavor for DA4ML (verilog/vhdl)')
    parser.add_argument('--latency-cutoff', '-lc', type=float, default=5, help='Latency cutoff for pipelining')
    parser.add_argument('--part-name', '-p', type=str, default='xcvu13p-flga2577-2-e', help='FPGA part name')
    parser.add_argument('--verbose', '-v', default=1, type=int, help='Set verbosity level (0: silent, 1: info, 2: debug)')
    parser.add_argument('--validate-rtl', '-vr', action='store_true', help='Validate RTL by Verilator (and GHDL)')
    parser.add_argument('--n-threads', '-j', type=int, default=4, help='Number of threads for compilation and DAIS simulation')
    parser.add_argument('--metadata', '-meta', type=str, default=None, help='Path to metadata JSON file to be included')
    parser.add_argument(
        '--hw-config',
        '-hc',
        type=int,
        nargs=3,
        metavar=('ACCUM_SIZE', 'ADDER_SIZE', 'CUTOFF'),
        default=[1, -1, -1],
        help='Size of accumulator and adder, and cutoff threshold during tracing. No need to modify unless you know what you are doing.',
    )
    parser.add_argument('--delay-constraint', '-dc', type=int, default=2, help='Delay constraint for each CMVM block')
    parser.add_argument(
        '--no-openmp',
        '--no-omp',
        action='store_true',
        help='Disable OpenMP in RTL simulation; no effect if --validate-rtl is not set',
    )
    parser.add_argument(
        '--inputs-kif',
        '-ikif',
        type=int,
        nargs=3,
        default=None,
        help='Input precision in KIF format (keep_neg, int bits, frac bits), if known.',
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert Keras model to da4ml RTL model with random input test vectors')
    _add_convert_args(parser)
    args = parser.parse_args()
    convert_main(args)
