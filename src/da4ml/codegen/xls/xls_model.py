from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from xls.raw import jit_fn_predict

from ...types import CombLogic, Pipeline, minimal_kif
from .xls_codegen import build_xls_function, build_xls_io_wrapper


class XLSModel:
    def __init__(
        self,
        logic: CombLogic | Pipeline,
        prj_name: str | None = None,
    ):
        self._solution = logic
        self._prj_name = prj_name or 'xls_model'
        self.built = False

    def _build(self):
        """Build the XLS IR function from the solution."""
        if isinstance(self._solution, Pipeline):
            raise NotImplementedError('Pipeline XLS codegen not yet supported')
        self._pkg, self._fn = build_xls_function(self._solution, self._prj_name)
        self._pkg.set_top(self._prj_name)
        wrapper_name = f'{self._prj_name}_wrapper'
        self._wrapper_pkg, self._wrapper_fn = build_xls_io_wrapper(self._solution, wrapper_name)
        self._wrapper_pkg.set_top(wrapper_name)
        self.built = True

    def jit(self) -> XLSModel:
        """Build the XLS IR and JIT-compile for execution."""
        if not self.built:
            self._build()
        self._jit = self._wrapper_fn.to_jit()
        return self

    def compile(
        self,
        path: str | Path | None = None,
        scheduling_options: str = '',
        codegen_flags: str = 'generator: GENERATOR_KIND_COMBINATIONAL',
        with_delay_model: bool = False,
    ) -> str:
        """Generate Verilog from the XLS IR via schedule_and_codegen.

        Returns the Verilog text. If path is provided, writes it there.
        """
        if not self.built:
            self._build()
        result = self._pkg.schedule_and_codegen(
            scheduling_options_textproto=scheduling_options,
            codegen_flags_textproto=codegen_flags,
            with_delay_model=with_delay_model,
        )
        verilog = result.get_verilog_text()
        if path is not None:
            p = Path(path)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(verilog)
        return verilog

    def predict(self, data: NDArray | Sequence[NDArray], n_threads: int = 0) -> NDArray[np.float64]:
        """Run the JIT-compiled model on input data.

        Parameters
        ----------
        data : NDArray | Sequence[NDArray]
            Input data. Shape is flattened; number of samples determined by size / inp_size.
        n_threads : int
            Unused (XLS JIT is single-threaded per call). Kept for API compatibility.

        Returns
        -------
        NDArray[np.float64]
            Output in shape (n_samples, output_size).
        """
        assert self._jit is not None, 'Call .jit() before .predict()'
        sol = self._solution

        if isinstance(data, Sequence):
            data = np.concatenate([a.reshape(a.shape[0], -1) for a in data], axis=-1)

        inp_size, out_size = sol.shape
        assert data.size % inp_size == 0

        inp_kifs = np.array([minimal_kif(qi) for qi in sol.inp_qint])
        k_in, i_in, f_in = np.max(inp_kifs, axis=0)
        max_inp_bw = int(k_in + i_in + f_in)

        inp_int = np.floor(data.ravel() * 2.0**f_in).astype(np.int64)

        # Run all samples through the JIT in C++
        out_int = jit_fn_predict(self._jit._raw, inp_int, max_inp_bw, inp_size, out_size)

        out_kifs = np.array([minimal_kif(qi) for qi in sol.out_qint])
        k, i, f = np.max(out_kifs, axis=0)
        a, b, c = 2.0 ** (k + i + f), k * 2.0 ** (i + f), 2.0**-f
        return ((out_int.reshape(-1, out_size) + b) % a - b) * c

    def __repr__(self):
        inp_size, out_size = self._solution.shape
        inp_kifs = [minimal_kif(qi) for qi in self._solution.inp_qint]
        out_kifs = [minimal_kif(qi) for qi in self._solution.out_qint]
        in_bits = sum(sum(k) for k in inp_kifs)
        out_bits = sum(sum(k) for k in out_kifs)
        jitted = 'JIT compiled' if self._jit is not None else 'not compiled'
        return f'XLSModel({self._prj_name}): {inp_size} ({in_bits} bits) -> {out_size} ({out_bits} bits), {jitted}'
