import ctypes
import json
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

from ...cmvm.types import _minimal_kif
from .. import hls

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
        clock_period: float = 5,
        clock_uncertainty: float = 0.1,
        io_delay_minmax: tuple[float, float] = (0.2, 0.4),
        namespace: str = 'comb_logic',
        inline_header: bool = True,
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
        self.__src_root = Path(hls.__file__).parent
        self._lib = None
        self._uuid = None
        self._namespace = namespace
        self._inline_static_header = inline_header

        if pragma is None:
            if self._flavor == 'vitis':
                self._pragma = (
                    '#pragma HLS ARRAY_PARTITION variable=model_inp complete',
                    '#pragma HLS ARRAY_PARTITION variable=model_out complete',
                    '#pragma HLS PIPELINE II=1',
                )
            else:
                self._pragma = ()
        else:
            self._pragma = tuple(pragma)

    def write(self, metadata: dict[str, str | float] | None = None):
        (self._path / 'sim').mkdir(parents=True, exist_ok=True)
        (self._path / 'model').mkdir(parents=True, exist_ok=True)
        (self._path / 'src/static').mkdir(parents=True, exist_ok=True)
        (self._path / 'utils').mkdir(parents=True, exist_ok=True)

        # Main logic and bridge
        template_def, bridge = hls_logic_and_bridge_gen(
            self._solution,
            self._prj_name,
            self._flavor,
            self._pragma,
            4,
            0,
            self._print_latency,
            namespace=self._namespace,
        )

        headers = ['#pragma once']
        if not self._inline_static_header:
            headers.append('#include "bitshift.hh"')

        namespace_open = f'namespace {self._namespace} {{\n' if self._namespace else ''
        namespace_close = f'\n}} // namespace {self._namespace}\n' if self._namespace else ''

        with open(self._path / f'src/{self._prj_name}.hh', 'w') as f:
            content = '\n'.join(headers)
            if self._inline_static_header:
                with open(self.__src_root / f'source/{self._flavor}_bitshift.hh') as ff:
                    bitshift_content = ff.read()
                bitshift_lines = bitshift_content.splitlines()
                bitshift_include = bitshift_lines[1]
                bitshift_content = '\n'.join(bitshift_lines[2:]).strip() + '\n'
                content += f'\n{bitshift_include}'
            else:
                bitshift_content = ''
            content += f'\n{namespace_open}\n{bitshift_content}\n{template_def};{namespace_close}'
            f.write(content)

        with open(self._path / f'sim/{self._prj_name}_bridge.cc', 'w') as f:
            f.write(bridge)

        # Emulation script and static files
        shutil.copy(self.__src_root / 'source/binder_util.hh', self._path / 'sim')
        shutil.copy(self.__src_root / 'source/build_binder.mk', self._path / 'sim')

        # Inline the only static header
        if not self._inline_static_header:
            shutil.copy(self.__src_root / f'source/{self._flavor}_bitshift.hh', self._path / 'src/static/bitshift.hh')
        if self._flavor == 'vitis':
            shutil.copytree(self.__src_root / 'source/ap_types', self._path / 'src/static/ap_types', dirs_exist_ok=True)
        else:
            pass

        # Dump the comb logic
        self._solution.save(self._path / 'model/comb.json')

        # Out-of-context top fn and its header
        inp_type, out_type = get_io_types(self._solution, self._flavor)
        n_in, n_out = len(self._solution.inp_qint), len(self._solution.out_qint)
        fn_signature = f'void {self._prj_name}_fn({inp_type} model_inp[{n_in}], {out_type} model_out[{n_out}])'

        pragma_str = '\n'.join(self._pragma)

        ooc_header_def = f"""#pragma once
#include "{self._prj_name}.hh"
{namespace_open}
{fn_signature};
{namespace_close}
"""
        with open(self._path / f'utils/{self._prj_name}_ooc.hh', 'w') as f:
            f.write(ooc_header_def)

        ooc_cpp_def = f"""
#include "{self._prj_name}_ooc.hh"

{namespace_open}
{fn_signature} {{
{pragma_str}
    {self._prj_name}<{inp_type}, {out_type}>(model_inp, model_out);
}}
{namespace_close}
"""
        with open(self._path / f'utils/{self._prj_name}_ooc.cc', 'w') as f:
            f.write(ooc_cpp_def)

        # Metadata
        _metadata = {
            'cost': self._solution.cost,
            'flavor': self._flavor,
            'part_name': self._part_name,
            'clock_period': self._clock_period,
            'clock_uncertainty': self._clock_uncertainty,
            'io_delay_min': self._io_delay_minmax[0],
            'io_delay_max': self._io_delay_minmax[1],
        }
        if metadata is not None:
            _metadata.update(metadata)

        with open(self._path / 'metadata.json', 'w') as f:
            json.dump(_metadata, f)

        # OOC Build scripts
        for path in (self.__src_root).glob('source/build_*_prj.tcl'):
            with open(path) as f:
                tcl = f.read()
            tcl = tcl.replace('$::env(PROJECT_NAME)', self._prj_name)
            tcl = tcl.replace('$::env(DEVICE)', self._part_name)
            tcl = tcl.replace('$::env(CLOCK_PERIOD)', str(self._clock_period))
            tcl = tcl.replace('$::env(CLOCK_UNCERTAINTY)', str(self._clock_uncertainty))
            with open(self._path / path.name, 'w') as f:
                f.write(tcl)

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
            for p in (self._path / 'sim').iterdir():
                if not p.is_dir() and m.match(p.name):
                    p.unlink()

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

        self._load_lib(self._uuid)

    def _load_lib(self, uuid: str | None = None):
        uuid = uuid if uuid is not None else self._uuid
        self._uuid = uuid
        lib_path = self._path / f'sim/lib{self._prj_name}_{uuid}.so'
        if not lib_path.exists():
            raise RuntimeError(f'Library {lib_path} does not exist')
        self._lib = ctypes.CDLL(str(lib_path))

    def compile(self, verbose=False, openmp=True, o3: bool = False, clean=True, metadata: dict[str, str | float] | None = None):
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
        metadata : dict[str, str | float] | None, optional
            Extra metadata to write to the model folder, by default None

        Raises
        ------
        RuntimeError
            If compilation fails
        """
        self.write(metadata)
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

        n_threads : int, optional
            Number of threads to use for inference. If 0, will use all available threads, or the value of
            the DA_DEFAULT_THREADS environment variable if set. If < 0, OpenMP will be disabled. Default is 0.
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

        if n_threads == 0:
            n_threads = int(os.environ.get('DA_DEFAULT_THREADS', 0))

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
