import ctypes
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from uuid import uuid4

import numpy as np
from numpy.typing import NDArray

from ... import codegen
from ...cmvm.types import CascadedSolution, Solution, _minimal_kif
from ...trace.pipeline import to_pipeline
from . import comb_binder_gen, comb_logic_gen, generate_io_wrapper, pipeline_binder_gen, pipeline_logic_gen


def get_io_kifs(sol: Solution | CascadedSolution):
    inp_kifs = tuple(zip(*map(_minimal_kif, sol.inp_qint)))
    out_kifs = tuple(zip(*map(_minimal_kif, sol.out_qint)))
    return np.array(inp_kifs, np.int8), np.array(out_kifs, np.int8)


class VerilogModel:
    def __init__(
        self,
        solution: Solution | CascadedSolution,
        prj_name: str,
        path: str | Path,
        latency_cutoff: int = -1,
        print_latency: bool = True,
        part_name: str = 'xcvu13p-flga2577-2-e',
        clock_period: int = 5,
        clock_uncertainty: float = 0.1,
        io_delay_minmax: tuple[float, float] = (0.2, 0.4),
        register_layers: int = 1,
    ):
        self._solution = solution
        self._path = Path(path)
        self._prj_name = prj_name
        self._latency_cutoff = latency_cutoff
        self._print_latency = print_latency
        self.__src_root = Path(codegen.__file__).parent
        self._part_name = part_name
        self._clock_period = clock_period
        self._clock_uncertainty = clock_uncertainty
        self._io_delay_minmax = io_delay_minmax
        self._register_layers = register_layers

        self._pipe = solution if isinstance(solution, CascadedSolution) else None
        if latency_cutoff > 0 and self._pipe is None:
            assert isinstance(solution, Solution)
            self._pipe = to_pipeline(solution, latency_cutoff, verbose=False)

        if self._pipe is not None:
            # get actual latency cutoff
            latency_cutoff = int(max(max(st.latency) / (i + 1) for i, st in enumerate(self._pipe.solutions)))
            self._latency_cutoff = latency_cutoff

        self._lib = None

    def write(self):
        self._path.mkdir(parents=True, exist_ok=True)
        if self._pipe is not None:  # Pipeline
            # Main logic
            codes = pipeline_logic_gen(self._pipe, self._prj_name, self._print_latency, register_layers=self._register_layers)
            for k, v in codes.items():
                with open(self._path / f'{k}.v', 'w') as f:
                    f.write(v)

            # Build script
            with open(self.__src_root / 'verilog/source/build_prj.tcl') as f:
                tcl = f.read()
            tcl = tcl.replace('${DEVICE}', self._part_name)
            tcl = tcl.replace('${PROJECT_NAME}', self._prj_name)
            with open(self._path / 'build_prj.tcl', 'w') as f:
                f.write(tcl)

            # XDC
            with open(self.__src_root / 'verilog/source/template.xdc') as f:
                xdc = f.read()
            xdc = xdc.replace('${CLOCK_PERIOD}', str(self._clock_period))
            xdc = xdc.replace('${UNCERTAINITY_SETUP}', str(self._clock_uncertainty))
            xdc = xdc.replace('${UNCERTAINITY_HOLD}', str(self._clock_uncertainty))
            xdc = xdc.replace('${DELAY_MAX}', str(self._io_delay_minmax[0]))
            xdc = xdc.replace('${DELAY_MIN}', str(self._io_delay_minmax[1]))
            with open(self._path / f'{self._prj_name}.xdc', 'w') as f:
                f.write(xdc)

            # C++ binder w/ verilog wrapper for uniform bw
            binder = pipeline_binder_gen(self._pipe, f'{self._prj_name}_wrapper', 1, self._register_layers)

            # Verilog IO wrapper (non-uniform bw to uniform one, clk passthrough)
            io_wrapper = generate_io_wrapper(self._pipe, self._prj_name, True)

            self._pipe.save(self._path / 'pipeline.json')
        else:  # Comb
            assert isinstance(self._solution, Solution)

            # Main logic
            code = comb_logic_gen(self._solution, self._prj_name, self._print_latency, '`timescale 1ns/1ps')
            with open(self._path / f'{self._prj_name}.v', 'w') as f:
                f.write(code)

            # Verilog IO wrapper (non-uniform bw to uniform one, no clk)
            io_wrapper = generate_io_wrapper(self._solution, self._prj_name, False)
            binder = comb_binder_gen(self._solution, f'{self._prj_name}_wrapper')

        with open(self._path / f'{self._prj_name}_wrapper.v', 'w') as f:
            f.write(io_wrapper)
        with open(self._path / f'{self._prj_name}_wrapper_binder.cc', 'w') as f:
            f.write(binder)

        # Common resource copy
        shutil.copy(self.__src_root / 'verilog/source/shift_adder.v', self._path)
        shutil.copy(self.__src_root / 'verilog/source/build_binder.mk', self._path)
        shutil.copy(self.__src_root / 'verilog/source/ioutils.hh', self._path)
        self._solution.save(self._path / 'model.json')
        with open(self._path / 'misc.json', 'w') as f:
            f.write(f'{{"cost": {self._solution.cost}}}')

    def _compile(self, verbose=False, openmp=True, o3: bool = False, clean=True):
        """Same as compile, but will not write to the library

        Parameters
        ----------
        verbose : bool, optional
            Verbose output, by default False
        openmp : bool, optional
            Enable openmp, by default True
        o3 : bool | None, optional
            Turn on -O3 flag, by default False
        clean : bool, optional
            Remove obsolete shared object files, by default True

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
        if o3:
            args.append('fast')

        if clean:
            m = re.compile(r'^lib.*[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\.so$')
            for p in self._path.iterdir():
                if not p.is_dir() and m.match(p.name):
                    p.unlink()

        try:
            r = subprocess.run(args, env=env, check=True, cwd=self._path, capture_output=not verbose)
        except subprocess.CalledProcessError as e:
            print(e.stderr.decode(), file=sys.stderr)
            print(e.stdout.decode(), file=sys.stdout)
            raise RuntimeError('Compilation failed!!') from e
        if r.returncode != 0:
            print(r.stderr.decode(), file=sys.stderr)
            print(r.stdout.decode(), file=sys.stderr)
            raise RuntimeError('Compilation failed!!')

        self._load_lib(self._uuid)

    def _load_lib(self, uuid: str | None = None):
        uuid = uuid if uuid is not None else self._uuid
        self._uuid = uuid
        lib_path = self._path / f'lib{self._prj_name}_wrapper_{uuid}.so'
        if not lib_path.exists():
            raise RuntimeError(f'Library {lib_path} does not exist')
        self._lib = ctypes.CDLL(str(lib_path))

    def compile(self, verbose=False, openmp=True, o3: bool = False):
        """Compile the generated code to a emulator for logic simulation.

        Parameters
        ----------
        verbose : bool, optional
            Verbose output, by default False
        openmp : bool, optional
            Enable openmp, by default True
        o3 : bool | None, optional
            Turn on -O3 flag, by default False

        Raises
        ------
        RuntimeError
            If compilation fails
        """
        self.write()
        self._compile(verbose=verbose, openmp=openmp, o3=o3)
        self._load_lib()

    def predict(self, data: NDArray[np.floating]):
        """Run the model on the input data.

        Parameters
        ----------
        data : NDArray[np.floating]
            Input data to the model. The shape is ignored, and the number of samples is
            determined by the size of the data.

        Returns
        -------
        NDArray[np.float64]
            Output of the model in shape (n_samples, output_size).
        """
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
        inp_data[:] = data.ravel() * 2.0 ** np.max(f_in)

        inp_buf = inp_data.ctypes.data_as(ctypes.POINTER(ctypes.c_int32))
        out_buf = out_data.ctypes.data_as(ctypes.POINTER(ctypes.c_int32))
        self._lib.inference(inp_buf, out_buf, n_sample)

        # Unscale the output int32 to recover fp values
        k, i, f = np.max(k_out), np.max(i_out), np.max(f_out)
        a, b, c = 2.0 ** (k + i + f), 2.0 ** (i + f), 2.0**-f
        return ((out_data.reshape(n_sample, out_size) + b) % a - b) * c

    def __repr__(self):
        inp_size, out_size = self._solution.shape
        cost = round(self._solution.cost)
        kifs_in, kifs_out = get_io_kifs(self._solution)
        in_bits, out_bits = np.sum(kifs_in), np.sum(kifs_out)
        if self._pipe is not None:
            n_stage = len(self._pipe[0])
            delay_suffix = '' if self._register_layers == 1 else f'x {self._register_layers} '
            lat_cutoff = self._latency_cutoff
            reg_bits = self._pipe.reg_bits
            spec = f"""Top Module: {self._prj_name}\n====================
{inp_size} ({in_bits} bits) -> {out_size} ({out_bits} bits)
{n_stage} {delay_suffix}stages @ max_delay={lat_cutoff}
Estimated cost: {cost} LUTs, {reg_bits} FFs"""

        else:
            spec = f"""Top Module: {self._prj_name}\n====================
{inp_size} ({in_bits} bits) -> {out_size} ({out_bits} bits)
combinational @ delay={self._solution.latency}
Estimated cost: {cost} LUTs"""

        is_compiled = self._lib is not None
        if is_compiled:
            openmp = 'with OpenMP' if self._lib.openmp_enabled else ''  # type: ignore
            spec += f'\nEmulator is compiled {openmp} ({self._uuid[-12:]})'
        else:
            spec += '\nEmulator is **not compiled**'
        return spec
