import ctypes
import json
import os
import re
import shutil
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
from numpy.typing import NDArray

from ...trace.pipeline import to_pipeline
from ...types import CombLogic, Pipeline, minimal_kif
from .. import rtl


def get_io_kifs(sol: CombLogic | Pipeline):
    inp_kifs = tuple(zip(*map(minimal_kif, sol.inp_qint)))
    out_kifs = tuple(zip(*map(minimal_kif, sol.out_qint)))
    return np.array(inp_kifs, np.int8), np.array(out_kifs, np.int8)


def binder_gen(csol: Pipeline | CombLogic, module_name: str, II: int = 1):
    k_in, i_in, f_in = zip(*map(minimal_kif, csol.inp_qint))
    k_out, i_out, f_out = zip(*map(minimal_kif, csol.out_qint))
    max_inp_bw = max(k_in) + max(i_in) + max(f_in)
    max_out_bw = max(k_out) + max(i_out) + max(f_out)
    if isinstance(csol, CombLogic):
        II = latency = 0
    else:
        latency = len(csol.solutions)

    n_in, n_out = csol.shape
    return f"""#include <cstddef>
#include "binder_util.hh"
#include "V{module_name}.h"

struct {module_name}_config {{
    static const size_t N_inp = {n_in};
    static const size_t N_out = {n_out};
    static const size_t max_inp_bw = {max_inp_bw};
    static const size_t max_out_bw = {max_out_bw};
    static const size_t II = {II};
    static const size_t latency = {latency};
    typedef V{module_name} dut_t;
}};

extern "C" {{
bool openmp_enabled() {{
    return _openmp;
}}

void inference(int32_t *c_inp, int32_t *c_out, size_t n_samples, size_t n_threads) {{
    batch_inference<{module_name}_config>(c_inp, c_out, n_samples, n_threads);
}}
}}
"""


class at_path:
    def __init__(self, path: str | Path):
        self._path = Path(path)
        self._orig_cwd = None

    def __enter__(self):
        self._orig_cwd = Path.cwd()
        os.chdir(self._path)

    def __exit__(self, exc_type, exc_value, traceback):
        os.chdir(self._orig_cwd)  # type: ignore


def canon_name(name: str) -> str:
    return re.sub(r'\W|^(?=\d)', '_', name)


def verilog_comb_logic_gen_xls(sol: CombLogic, fn_name: str, print_latency: bool = False, timescale: str | None = None):
    from ..xls.xls_codegen import build_xls_function

    pkg, fn = build_xls_function(sol, fn_name)
    result = pkg.schedule_and_codegen(generator='combinational', output_port_name='model_out')
    verilog = result.get_verilog_text()
    if timescale is not None:
        verilog = f'{timescale}\n\n' + verilog
    return verilog


class RTLModel:
    def __init__(
        self,
        logic: CombLogic | Pipeline,
        path: str | Path,
        prj_name: str | None = None,
        flavor: str = 'verilog',
        n_stages: int = -1,
        latency_cutoff: float = -1,
        print_latency: bool = True,
        part_name: str = 'xcvu13p-flga2577-2-e',
        clock_period: float = 5,
        clock_uncertainty: float = 0.1,
        io_delay_minmax: tuple[float, float] = (0.2, 0.4),
    ):
        """Class for generating RTL code and emulator from a CombLogic or Pipeline solution.

        Parameters
        ----------
        solution : CombLogic | Pipeline
            The logic object represented in DAIS to be converted to RTL.
        path : str | Path
            The path to save the generated RTL project. The directory will be created if it does not exist.
        prj_name : str | None, optional
            Top module name of the generated RTL. If None, canonicalized name of the directory will be used. Default is None.
        flavor : str, optional
            Either 'verilog' or 'vhdl'. Default is 'verilog'.
        n_stages : int, optional
            Number of pipeline stages of the generated RTL. If negative, it will be determined by latency_cutoff.
            If both n_stages and latency_cutoff are not specified, the generated RTL will be purely combinational. Default is -1.
        latency_cutoff : float, optional
            Latency cutoff for determining the number of stages. Only used if n_stages is -1. Default is -1.
        print_latency : bool, optional
            Whether to print the latency of each stage in the generated RTL. Default is True.
        part_name : str, optional
            Device part name used in default OOC scripts. Default is 'xcvu13p-flga2577-2-e'.
        clock_period : float, optional
            Clock period in ns used in default OOC scripts. Default is 5.
        clock_uncertainty : float, optional
            Clock uncertainty in ns used in default OOC scripts. Default is 0.1.
        io_delay_minmax : tuple[float, float], optional
            Minimum and maximum IO delay in ns used in default OOC scripts. Default is (0.2, 0.4).
        """
        self._flavor = flavor.lower()
        self._solution = logic
        self._path = Path(path).resolve()
        self._prj_name = prj_name or canon_name(self._path.stem)
        self._n_stages = n_stages
        self._latency_cutoff = latency_cutoff
        self._print_latency = print_latency
        self.__src_root = Path(rtl.__file__).parent
        self._part_name = part_name
        self._clock_period = clock_period
        self._clock_uncertainty = clock_uncertainty
        self._io_delay_minmax = io_delay_minmax
        self._place_holder = False

        assert self._flavor in ('vhdl', 'verilog'), f'Unsupported flavor {flavor}, only vhdl and verilog are supported.'

        self._pipe = logic if isinstance(logic, Pipeline) else None

        if (latency_cutoff > 0 or n_stages > 0) and self._pipe is None:
            _latency_cutoff = latency_cutoff if latency_cutoff > 0 else None
            _n_stages = n_stages if n_stages > 0 else None
            assert isinstance(logic, CombLogic)
            self._pipe = to_pipeline(logic, _n_stages, _latency_cutoff, verbose=False)

        if self._pipe is not None:
            # get actual latency cutoff
            latency_cutoff = max(max(st.latency) / (i + 1) for i, st in enumerate(self._pipe.solutions))
            self._latency_cutoff = latency_cutoff

        self._lib = None
        self._uuid = None

    def write(self, metadata: None | dict[str, Any] = None, xls_opt: bool = False):
        """Write the RTL project to the specified path.

        Parameters
        ----------
        metadata : dict[str, Any] | None, optional
            Additional metadata to write to `metadata.json`, by default None
        xls_opt : bool, optional
            Whether to apply XLS optimizations to the generated RTL.
            Requires `xls-python` package, only applicable to verilog codegen. Default is False.
        """

        flavor = self._flavor
        suffix = 'v' if flavor == 'verilog' else 'vhd'
        if flavor == 'vhdl':
            assert not xls_opt, 'XLS optimizations are not supported for VHDL codegen.'
            from .vhdl import comb_logic_gen, generate_io_wrapper, pipeline_logic_gen
        else:  # verilog
            from .verilog import generate_io_wrapper, pipeline_logic_gen

            if xls_opt:
                comb_logic_gen = verilog_comb_logic_gen_xls
            else:
                from .verilog import comb_logic_gen

        from .verilog.comb import table_mem_gen

        (self._path / 'src/static').mkdir(parents=True, exist_ok=True)
        (self._path / 'sim').mkdir(exist_ok=True)
        (self._path / 'model').mkdir(exist_ok=True)
        (self._path / 'src/memfiles').mkdir(exist_ok=True)

        # Build scripts
        for path in (self.__src_root).glob('common_source/build_*_prj.tcl'):
            with open(path) as f:
                tcl = f.read()
            tcl = tcl.replace('$::env(DEVICE)', self._part_name)
            tcl = tcl.replace('$::env(PROJECT_NAME)', self._prj_name)
            tcl = tcl.replace('$::env(SOURCE_TYPE)', flavor)
            with open(self._path / path.name, 'w') as f:
                f.write(tcl)

        if isinstance(self._solution, CombLogic):
            self._solution.save(self._path / 'model/comb.json')

        if self._pipe is not None:  # Pipeline
            self._pipe.save(self._path / 'model/pipeline.json')

            if not self._place_holder:
                # Main logic
                codes = pipeline_logic_gen(
                    csol=self._pipe,
                    name=self._prj_name,
                    print_latency=self._print_latency,
                    comb_logic_gen_fn=comb_logic_gen,
                )

                # Table memory files
                memfiles: dict[str, str] = {}
                for _comb in self._pipe.solutions:
                    memfiles.update(table_mem_gen(_comb))

                for k, v in codes.items():
                    with open(self._path / f'src/{k}.{suffix}', 'w') as f:
                        f.write(v)
            else:
                memfiles = {}

            # Timing constraint
            for fmt in ('xdc', 'sdc'):
                with open(self.__src_root / f'common_source/template.{fmt}') as f:
                    constraint = f.read()
                constraint = constraint.replace('$::env(CLOCK_PERIOD)', str(self._clock_period))
                constraint = constraint.replace('$::env(UNCERTAINITY_SETUP)', str(self._clock_uncertainty))
                constraint = constraint.replace('$::env(UNCERTAINITY_HOLD)', str(self._clock_uncertainty))
                constraint = constraint.replace('$::env(DELAY_MAX)', str(self._io_delay_minmax[1]))
                constraint = constraint.replace('$::env(DELAY_MIN)', str(self._io_delay_minmax[0]))
                with open(self._path / f'src/{self._prj_name}.{fmt}', 'w') as f:
                    f.write(constraint)

            # C++ binder w/ HDL wrapper for uniform bw
            binder = binder_gen(self._pipe, f'{self._prj_name}_wrapper', 1)

            # Verilog IO wrapper (non-uniform bw to uniform one, clk passthrough)
            io_wrapper = generate_io_wrapper(self._pipe, self._prj_name, True)
        else:  # Comb
            assert isinstance(self._solution, CombLogic)

            if not self._place_holder:
                # Table memory files
                memfiles = table_mem_gen(self._solution)

                # Main logic
                code = comb_logic_gen(self._solution, self._prj_name, self._print_latency, '`timescale 1ns/1ps')
                with open(self._path / f'src/{self._prj_name}.{suffix}', 'w') as f:
                    f.write(code)
            else:
                memfiles = {}

            # Verilog IO wrapper (non-uniform bw to uniform one, no clk)
            io_wrapper = generate_io_wrapper(self._solution, self._prj_name, False)
            binder = binder_gen(self._solution, f'{self._prj_name}_wrapper')

        # Write table memory files
        for name, mem in memfiles.items():
            with open(self._path / 'src/memfiles' / name, 'w') as f:
                f.write(mem)

        with open(self._path / f'src/{self._prj_name}_wrapper.{suffix}', 'w') as f:
            f.write(io_wrapper)
        with open(self._path / f'sim/{self._prj_name}_wrapper_binder.cc', 'w') as f:
            f.write(binder)

        # Common resource copy
        for path in self.__src_root.glob(f'{flavor}/source/*.{suffix}'):
            shutil.copy(path, self._path / 'src/static')

        shutil.copy(self.__src_root / 'common_source/build_binder.mk', self._path / 'sim')
        shutil.copy(self.__src_root / 'common_source/ioutil.hh', self._path / 'sim')
        shutil.copy(self.__src_root / 'common_source/binder_util.hh', self._path / 'sim')

        _metadata = {
            'cost': self._solution.cost,
            'flavor': self._flavor,
            'part_name': self._part_name,
        }
        _comb = self._solution if isinstance(self._solution, CombLogic) else self._solution.solutions[0]
        _metadata['adder_size'] = _comb.adder_size
        _metadata['carry_size'] = _comb.carry_size
        if self._pipe is not None:
            _metadata['latency'] = len(self._pipe[0])
            _metadata['reg_bits'] = self._pipe.reg_bits
            _metadata['clock_period'] = self._clock_period
            _metadata['latency_cutoff'] = self._latency_cutoff
            _metadata['clock_period'] = self._clock_period
            _metadata['clock_uncertainty'] = self._clock_uncertainty
            _metadata['io_delay_min'] = self._io_delay_minmax[0]
            _metadata['io_delay_max'] = self._io_delay_minmax[1]

        if metadata is not None:
            _metadata.update({k: v for k, v in metadata.items() if k not in _metadata})

        with open(self._path / 'metadata.json', 'w') as f:
            json.dump(_metadata, f)

    def _compile(self, verbose=False, openmp=True, nproc=None, o3: bool = False, clean=True, _env: dict[str, str] | None = None):
        """Same as compile, but will not write to the library

        Parameters
        ----------
        verbose : bool, optional
            Verbose output, by default False
        openmp : bool, optional
            Enable openmp, by default True
        nproc : int | None, optional
            Number of processes to use for compilation, by default None
            If None, will use the number of CPU cores, but not more than 32.
        o3 : bool | None, optional
            Turn on -O3 flag, by default False
        clean : bool, optional
            Remove obsolete shared object files and `obj_dir`, by default True

        Raises
        ------
        RuntimeError
            If compilation fails
        """

        self._uuid = str(uuid4())
        args = ['make', '-f', 'build_binder.mk']
        env = os.environ.copy()
        env['VM_PREFIX'] = f'{self._prj_name}_wrapper'
        env['STAMP'] = self._uuid
        env['EXTRA_CXXFLAGS'] = '-fopenmp' if openmp else ''
        env['VERILATOR_FLAGS'] = '-Wall' if self._flavor == 'verilog' else ''
        if _env is not None:
            env.update(_env)
        if nproc is not None:
            env['N_JOBS'] = str(nproc)
        if o3:
            args.append('fast')

        if clean:
            m = re.compile(r'^lib.*[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\.so$')
            for p in (self._path / 'sim').iterdir():
                if not p.is_dir() and m.match(p.name):
                    p.unlink()
            subprocess.run(
                ['make', '-f', 'build_binder.mk', 'clean'],
                env=env,
                cwd=self._path / 'sim',
                check=True,
                capture_output=not verbose,
            )

        try:
            r = subprocess.run(args, env=env, check=True, cwd=self._path / 'sim', capture_output=not verbose)
        except subprocess.CalledProcessError as e:
            print(e.stderr.decode(), file=sys.stderr)
            print(e.stdout.decode(), file=sys.stdout)
            raise RuntimeError('Compilation failed!!') from e
        if r.returncode != 0:
            print(r.stderr.decode(), file=sys.stderr)
            print(r.stdout.decode(), file=sys.stderr)
            raise RuntimeError('Compilation failed!!')

        if clean:
            subprocess.run(['rm', '-rf', 'obj_dir'], cwd=self._path / 'sim', check=True, capture_output=not verbose)

        self._load_lib(self._uuid)

    def _load_lib(self, uuid: str | None = None):
        uuid = uuid if uuid is not None else self._uuid
        if uuid is None:
            # load .so if there is only one, otherwise raise an error
            libs = list(self._path.glob(f'sim/lib{self._prj_name}_wrapper_*.so'))
            if len(libs) == 0:
                raise RuntimeError(f'Cannot load library, found {len(libs)} libraries in {self._path}')
            uuid = libs[0].name.split('_')[-1].split('.', 1)[0]
        self._uuid = uuid
        lib_path = self._path / f'sim/lib{self._prj_name}_wrapper_{uuid}.so'
        if not lib_path.exists():
            raise RuntimeError(f'Library {lib_path} does not exist')
        self._lib = ctypes.CDLL(str(lib_path))

    def compile(
        self,
        verbose=False,
        openmp=True,
        nproc: int | None = None,
        o3: bool = False,
        clean=True,
        metadata: None | dict[str, Any] = None,
        xls_opt: bool = False,
    ):
        """Compile the generated code to a emulator for logic simulation.

        Parameters
        ----------
        verbose : bool, optional
            Verbose output, by default False
        openmp : bool, optional
            Enable openmp, by default True
        nproc : int | None, optional
            Number of processes to use for compilation, by default None
            If None, will use the number of CPU cores, but not more than 32.
        o3 : bool | None, optional
            Turn on -O3 flag, by default False
        clean : bool, optional
            Remove obsolete shared object files and `obj_dir`, by default True
        metadata : dict[str, Any] | None, optional
            Additional metadata to write to `metadata.json`, by default None
        xls_opt : bool, optional
            Whether to apply XLS optimizations to the generated RTL.
            Requires `xls-python` package, only applicable to verilog codegen. Default is False.

        Raises
        ------
        RuntimeError
            If compilation fails
        """

        self.write(metadata=metadata, xls_opt=xls_opt)
        self._compile(verbose=verbose, openmp=openmp, nproc=nproc, o3=o3, clean=clean)

    def predict(self, data: NDArray | Sequence[NDArray], n_threads: int = 0) -> NDArray[np.float32]:
        """Run the model on the input data.

        Parameters
        ----------
        data : NDArray[np.floating]|Sequence[NDArray[np.floating]]
            Input data to the model. The shape is ignored, and the number of samples is
            determined by the size of the data.

        Returns
        -------
        NDArray[np.float64]
            Output of the model in shape (n_samples, output_size).
        n_threads : int, optional
            Number of threads to use for inference. If 0, will use all available threads, or the value of
            the DA_DEFAULT_THREADS environment variable if set. If < 0, OpenMP will be disabled. Default is 0.
        """

        if isinstance(data, Sequence):
            data = np.concatenate([a.reshape(a.shape[0], -1) for a in data], axis=-1)

        assert self._lib is not None, 'Library not loaded, call .compile() first.'
        inp_size, out_size = self._solution.shape

        assert data.size % inp_size == 0, f'Input size {data.size} is not divisible by {inp_size}'
        n_sample = data.size // inp_size

        kifs_in, kifs_out = get_io_kifs(self._solution)
        k_in, i_in, f_in = map(np.max, kifs_in)
        k_out, i_out, f_out = map(np.max, kifs_out)
        assert k_in + i_in + f_in <= 32, "Padded inp bw doesn't fit in int32. Emulation not supported"
        assert k_out + i_out + f_out <= 32, "Padded out bw doesn't fit in int32. Emulation not supported"

        inp_data = np.empty(n_sample * inp_size, dtype=np.int32)
        out_data = np.empty(n_sample * out_size, dtype=np.int32)

        # Convert to int32 matching the LSB position
        inp_data[:] = np.floor(data.ravel() * 2.0**f_in)

        inp_buf = inp_data.ctypes.data_as(ctypes.POINTER(ctypes.c_int32))
        out_buf = out_data.ctypes.data_as(ctypes.POINTER(ctypes.c_int32))

        if n_threads == 0:
            n_threads = int(os.environ.get('DA_DEFAULT_THREADS', 0))

        with at_path(self._path / 'src/memfiles'):
            self._lib.inference(inp_buf, out_buf, n_sample, n_threads)

        # Unscale the output int32 to recover fp values
        k, i, f = np.max(k_out), np.max(i_out), np.max(f_out)
        a, b, c = 2.0 ** (k + i + f), k * 2.0 ** (i + f), 2.0**-f
        return ((out_data.reshape(n_sample, out_size) + b) % a - b) * c.astype(np.float32)

    def __repr__(self):
        inp_size, out_size = self._solution.shape
        cost = round(self._solution.cost)
        kifs_in, kifs_out = get_io_kifs(self._solution)
        in_bits, out_bits = np.sum(kifs_in), np.sum(kifs_out)
        if self._pipe is not None:
            n_stage = len(self._pipe[0])
            lat_cutoff = self._latency_cutoff
            reg_bits = self._pipe.reg_bits
            spec = f"""Top Module: {self._prj_name}\n====================
{inp_size} ({in_bits} bits) -> {out_size} ({out_bits} bits)
{n_stage} stages @ max_delay={lat_cutoff}
Estimated cost: {cost} LUTs, {reg_bits} FFs"""

        else:
            spec = f"""Top Module: {self._prj_name}\n====================
{inp_size} ({in_bits} bits) -> {out_size} ({out_bits} bits)
combinational @ delay={self._solution.latency}
Estimated cost: {cost} LUTs"""

        is_compiled = self._lib is not None
        if is_compiled:
            assert self._uuid is not None
            openmp = 'with OpenMP' if self._lib.openmp_enabled() else ''  # type: ignore
            spec += f'\nEmulator is compiled {openmp} ({self._uuid[-12:]})'
        else:
            spec += '\nEmulator is **not compiled**'
        return spec
