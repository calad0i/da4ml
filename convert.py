import json
from pathlib import Path

import hgq  # noqa: F401
import keras
import numpy as np


def to_hls4ml(model: keras.Model, da: bool, path: Path, n_test_sample: int, period: float, unc: float):
    from hls4ml.converters import convert_from_keras_model

    hls_config = {
        'Model': {
            'Precision': 'ap_fixed<-1,0>',
            'ReuseFactor': 1,
            'Strategy': 'distributed_arithmetic' if da else 'latency',
        },
    }
    model_hls = convert_from_keras_model(
        model,
        hls_config=hls_config,
        output_dir=str(path),
        clock_period=period,
        clock_uncertainty=f'{unc}%',
    )
    model_hls.write()
    if n_test_sample:
        data_in = [np.random.rand(n_test_sample, *inp.shape[1:]).astype(np.float32) * 64 - 32 for inp in model.inputs]
        if len(data_in) == 1:
            data_in = data_in[0]
        y_keras = model.predict(data_in, batch_size=9999999, verbose=0)  # type: ignore
        model_hls._compile()
        y_hls = model_hls.predict(data_in)
        if isinstance(y_hls, list):
            y_hls = np.concatenate([y.ravel() for y in y_hls])
            y_keras = np.concatenate([y.ravel() for y in y_keras])
        else:
            y_hls = y_hls.ravel()
            y_keras = y_keras.ravel()
        mask = y_hls != y_keras
        if np.any(mask):
            ndiff = np.sum(mask)
            n = y_hls.size
            n_nonzero = np.sum(y_keras != 0)
            maxdiff = np.max(np.abs(y_hls - y_keras))
            print(f'[WARNING] {ndiff}/{n} ({n_nonzero}) mismatches (maxdiff={maxdiff})')


def to_da4ml(model: keras.Model, path: Path, n_test_sample: int, period: float, unc: float, flavor: str, verbose: bool = False):
    from da4ml.codegen import RTLModel
    from da4ml.converter import trace_model
    from da4ml.trace import HWConfig, comb_trace

    inp, out = trace_model(model, HWConfig(1, -1, -1), {'hard_dc': 2}, verbose)
    comb = comb_trace(inp, out)
    rtl_model = RTLModel(comb, 'model', path, flavor, period * 1, True, clock_uncertainty=unc / 100, clock_period=period)
    rtl_model.write()
    if verbose:
        print('Model written')
    if n_test_sample:
        data_in = [np.random.rand(n_test_sample, *inp.shape[1:]).astype(np.float32) * 64 - 32 for inp in model.inputs]
        y_keras = model.predict(data_in, batch_size=16384, verbose=0)  # type: ignore
        y_comb = comb.predict(data_in, n_threads=2)
        if isinstance(y_keras, list):
            y_keras = np.concatenate([y.reshape(n_test_sample, -1) for y in y_keras], axis=1)
        else:
            y_keras = y_keras.reshape(n_test_sample, -1)
        mask = y_comb != y_keras

        total = y_comb.size
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

        # if verbose:
        #     print('Verilating...')
        # for _ in range(3):
        #     try:
        #         rtl_model._compile(nproc=4)
        #         break
        #     except RuntimeError:
        #         pass
        # y_da4ml = rtl_model.predict(data_in)
        # if not np.all(y_comb == y_da4ml):
        #     print('[CRITICAL ERROR] Mismatch between traced model and DA4ML model!')


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('model', type=Path, help='Path to the Keras model file (HDF5 or SavedModel)')
    parser.add_argument('outdir', type=Path, help='Output directory')
    parser.add_argument('--framework', '-f', choices=['hls4ml', 'da4ml'], help='Framework to use for conversion', required=True)
    parser.add_argument('--da', action='store_true', help='Use distributed arithmetic (only for hls4ml)')
    parser.add_argument('--n_test_sample', type=int, default=131072, help='Number of test samples for validation')
    parser.add_argument('--period', type=float, default=5.0, help='Clock period in ns')
    parser.add_argument('--unc', type=float, default=10.0, help='Clock uncertainty in percent')
    parser.add_argument('--flavor', type=str, default='verilog', help='Flavor for DA4ML (verilog/vhdl)')
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose output')
    args = parser.parse_args()

    model: keras.Model = keras.models.load_model(args.model)  # type: ignore
    if args.verbose:
        model.summary()
    args.outdir.mkdir(parents=True, exist_ok=True)

    if args.framework == 'hls4ml':
        to_hls4ml(model, args.da, args.outdir, args.n_test_sample, args.period, args.unc)
    else:
        to_da4ml(model, args.outdir, args.n_test_sample, args.period, args.unc, flavor=args.flavor, verbose=args.verbose)
