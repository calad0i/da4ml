import ctypes
import os
import re
import shutil
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import TypeVar
from uuid import uuid4

import numpy as np
from numpy.typing import NDArray

from da4ml.cmvm.types import CombLogic
from da4ml.codegen.hls.hls_codegen import get_io_types, hls_logic_and_bridge_gen

from ... import codegen
from ...cmvm.types import _minimal_kif

T = TypeVar('T', bound=np.floating)


class HLSModel:
    def __init__(
        self,
        solution: CombLogic,
        prj_name: str,
        path: str | Path,
        flavor: str = 'vitis',
        print_latency: bool = True,
        part_name: str = 'xcvu13p-flga2577-2-e',
        pragma: Sequence[str] | None = None,
        clock_period: int = 5,
        clock_uncertainty: float = 0.1,
        io_delay_minmax: tuple[float, float] = (0.2, 0.4),
    ):
        self._solution = solution
        self._prj_name = prj_name
        self._path = Path(path).resolve()
        self._flavor = flavor.lower()
        assert self._flavor in ('vitis', 'hlslib', 'oneapi'), f'Unsupported HLS flavor: {self._flavor}'
        self._print_latency = print_latency
        self._part_name = part_name
        self._clock_period = clock_period
        self._clock_uncertainty = clock_uncertainty
        self._io_delay_minmax = io_delay_minmax
        self.__src_root = Path(codegen.__file__).parent
        self._lib = None
        self._uuid = None

        if pragma is None:
            if self._flavor == 'vitis':
                self._pragma = (
                    '#pragma HLS ARRAY_PARTITION variable=inp complete',
                    '#pragma HLS ARRAY_PARTITION variable=out complete',
                    '#pragma HLS PIPELINE II=1',
                )
            else:
                self._pragma = ()
        else:
            self._pragma = tuple(pragma)

    def write(self):
        if not self._path.exists():
            self._path.mkdir(parents=True, exist_ok=True)
        template_def, bridge = hls_logic_and_bridge_gen(
            self._solution,
            self._prj_name,
            self._flavor,
            ['#pragma HLS INLINE'],
            4,
            0,
            self._print_latency,
        )

        headers = ['#pragma once', '#include "bitshift.hh"']

        inp_type, out_type = get_io_types(self._solution, self._flavor)
        n_in, n_out = len(self._solution.inp_qint), len(self._solution.out_qint)
        template_signature = (
            f'template <typename inp_t, typename out_t>\nvoid {self._prj_name}(inp_t inp[{n_in}], out_t out[{n_out}]);'
        )
        fn_signature = f'void {self._prj_name}_fn({inp_type} inp[{n_in}], {out_type} out[{n_out}])'

        with open(self._path / f'{self._prj_name}.hh', 'w') as f:
            f.write('\n'.join(headers) + '\n\n')
            f.write(f'{template_signature}\n\n{fn_signature};\n')

        pragma_str = '\n'.join(self._pragma)
        cpp_def = f"""
#include "{self._prj_name}.hh"

{template_def}

{fn_signature} {{
{pragma_str}
    {self._prj_name}<{inp_type}, {out_type}>(inp, out);
}}
"""
        with open(self._path / f'{self._prj_name}.cc', 'w') as f:
            f.write(cpp_def)

        with open(self._path / f'{self._prj_name}_bridge.cc', 'w') as f:
            f.write(bridge)

        shutil.copy(self.__src_root / 'hls/source/binder_util.hh', self._path)
        shutil.copy(self.__src_root / f'hls/source/{self._flavor}_bitshift.hh', self._path / 'bitshift.hh')
        shutil.copy(self.__src_root / 'hls/source/build_binder.mk', self._path)
        if self._flavor == 'vitis':
            shutil.copytree(self.__src_root / 'hls/source/ap_types', self._path / 'ap_types', dirs_exist_ok=True)
        else:
            pass

        self._solution.save(self._path / 'project.json')

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
        env['PRJ_NAME'] = self._prj_name
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
        lib_path = self._path / f'lib{self._prj_name}_{uuid}.so'
        if not lib_path.exists():
            raise RuntimeError(f'Library {lib_path} does not exist')
        self._lib = ctypes.CDLL(str(lib_path))

    def compile(self, verbose=False, openmp=True, o3: bool = False, clean=True):
        """Compile the model to a shared object file

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
        self.write()
        self._compile(verbose, openmp, o3, clean)

    def predict(self, data: NDArray[T] | Sequence[NDArray[T]], n_threads: int = 0) -> NDArray[T]:
        """Run the model on the input data.

        Parameters
        ----------
        data: NDArray[np.floating] | Sequence[NDArray[np.floating]]
            Input data to the model. The shape is ignored, and the number of samples is
            determined by the size of the data.

        Returns
        -------
        NDArray[np.floating]
            Output of the model in shape (n_samples, output_size).
        """
        assert self._lib is not None, 'Library not loaded, call .compile() first.'
        inp_size, out_size = self._solution.shape

        if isinstance(data, Sequence):
            data = np.concatenate([a.reshape(a.shape[0], -1) for a in data], axis=-1)

        dtype = data.dtype
        if dtype not in (np.float32, np.float64):
            raise TypeError(f'Unsupported input data type: {dtype}. Expected float32 or float64.')
        c_dtype = ctypes.c_float if dtype == np.float32 else ctypes.c_double

        assert data.size % inp_size == 0, f'Input size {data.size} is not divisible by {inp_size}'
        n_sample = data.size // inp_size

        inp_data = np.ascontiguousarray(data)
        out_data = np.empty(n_sample * out_size, dtype=dtype)

        inp_buf = inp_data.ctypes.data_as(ctypes.POINTER(c_dtype))
        out_buf = out_data.ctypes.data_as(ctypes.POINTER(c_dtype))
        if dtype == np.float32:
            self._lib.inference_f32(inp_buf, out_buf, n_sample, n_threads)
        else:
            self._lib.inference_f64(inp_buf, out_buf, n_sample, n_threads)

        return out_data.reshape(n_sample, out_size)  # type: ignore

    def __repr__(self):
        inp_size, out_size = self._solution.shape
        inp_size, out_size = self._solution.shape
        cost = round(self._solution.cost)
        inp_kifs = tuple(zip(*map(_minimal_kif, self._solution.inp_qint)))
        out_kifs = tuple(zip(*map(_minimal_kif, self._solution.out_qint)))
        in_bits, out_bits = np.sum(inp_kifs), np.sum(out_kifs)

        spec = f"""Top Function: {self._prj_name}\n====================
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
