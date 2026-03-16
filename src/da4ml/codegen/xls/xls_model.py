from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from xls import Bits, Value

from ...types import CombLogic, Pipeline, minimal_kif
from .xls_codegen import build_xls_function


class XLSModel:
    def __init__(
        self,
        logic: CombLogic | Pipeline,
        prj_name: str | None = None,
    ):
        self._solution = logic
        self._prj_name = prj_name or 'xls_model'
        # self._pkg = None
        # self._fn = None
        # self._jit = None
        self.built = False

    def _build(self):
        """Build the XLS IR function from the solution."""
        if isinstance(self._solution, Pipeline):
            raise NotImplementedError('Pipeline XLS codegen not yet supported')
        self._pkg, self._fn = build_xls_function(self._solution, self._prj_name)
        self._pkg.set_top(self._prj_name)
        self._built = True

    def jit(self) -> XLSModel:
        """Build the XLS IR and JIT-compile for execution."""
        if not self.built:
            self._build()
        self._jit = self._fn.to_jit()
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
        if self._fn is None:
            self._build()
        result = self._pkg.schedule_and_codegen(
            scheduling_options=scheduling_options,
            codegen_flags=codegen_flags,
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
        n_samples = data.size // inp_size
        data_2d = data.reshape(n_samples, inp_size)

        # Compute input KIF and bit layout
        inp_kifs_list = [minimal_kif(qi) for qi in sol.inp_qint]
        inp_bws = [k + i + f for k, i, f in inp_kifs_list]
        inp_fracs = [f for _, _, f in inp_kifs_list]
        total_inp_bits = sum(inp_bws)

        # Output KIF from out_qint (kernel handles shifts/negs, predict just scales)
        out_kifs_list = [minimal_kif(qi) for qi in sol.out_qint]
        out_bws = [k + i + f for k, i, f in out_kifs_list]
        out_fracs = [f for _, _, f in out_kifs_list]
        out_signeds = [bool(k) for k, _, _ in out_kifs_list]
        total_out_bits = sum(out_bws)

        results = np.empty((n_samples, out_size), dtype=np.float64)

        for s in range(n_samples):
            # Pack input: convert each element to fixed-point int, concatenate bits
            packed = 0
            bit_pos = 0
            for j in range(inp_size):
                bw = inp_bws[j]
                if bw == 0:
                    continue
                frac = inp_fracs[j]
                int_val = int(np.floor(data_2d[s, j] * (2.0**frac)))
                # Mask to bw bits (2's complement)
                int_val = int_val & ((1 << bw) - 1)
                packed |= int_val << bit_pos
                bit_pos += bw

            # Create XLS value and run
            if total_inp_bits <= 64:
                inp_val = Value.make_ubits(total_inp_bits, packed)
            else:
                byte_len = (total_inp_bits + 7) // 8
                inp_val = Value.from_bits(Bits.from_bytes(total_inp_bits, packed.to_bytes(byte_len, 'little')))

            out_val = self._jit.run([inp_val])

            # Unpack output
            if total_out_bits <= 64:
                out_packed = out_val.get_bits().to_uint64()
            else:
                out_bytes = out_val.get_bits().to_bytes()
                out_packed = int.from_bytes(out_bytes, 'little')

            bit_pos = 0
            for j in range(out_size):
                obw = out_bws[j]
                if obw == 0:
                    results[s, j] = 0.0
                    bit_pos += obw
                    continue
                raw_int = (out_packed >> bit_pos) & ((1 << obw) - 1)
                # Sign extend if signed
                if out_signeds[j] and raw_int >= (1 << (obw - 1)):
                    raw_int -= 1 << obw
                # Convert to float (kernel already applied shifts/negs)
                frac = out_fracs[j]
                results[s, j] = raw_int * (2.0**-frac)
                bit_pos += obw

        return results

    def __repr__(self):
        inp_size, out_size = self._solution.shape
        inp_kifs = [minimal_kif(qi) for qi in self._solution.inp_qint]
        out_kifs = [minimal_kif(qi) for qi in self._solution.out_qint]
        in_bits = sum(sum(k) for k in inp_kifs)
        out_bits = sum(sum(k) for k in out_kifs)
        jitted = 'JIT compiled' if self._jit is not None else 'not compiled'
        return f'XLSModel({self._prj_name}): {inp_size} ({in_bits} bits) -> {out_size} ({out_bits} bits), {jitted}'
