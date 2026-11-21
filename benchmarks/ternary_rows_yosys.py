#!/usr/bin/env python3
"""Yosys mapping table for the opt-in ternary_rows Verilog option.

Run with da4ml (patched) importable and Yosys on PATH or given by --yosys:
python benchmarks/ternary_rows_yosys.py --count 50 --size 8 --dc 2 --out ternary_8x8.csv

For each seeded matrix of two families (integers in [129, 255], the convention of the da4ml paper's random-matrix
benchmark; and integers in [-8, 7]), with 8-bit signed inputs, the combinational Verilog is written twice from the
same solution: ternary_rows=False (shift_adder) and ternary_rows=True.  Each is mapped with
`synth_xilinx -family xcup -nodsp -flatten`.  Columns per variant:
  lut         LUT1..LUT6 + LUT6_2 cells
  inv         INV cells
  carry4, carry8   carry cells (Yosys may map inferred adders to CARRY4 even for xcup)
  ltp         Yosys `ltp -noff`: cells on the longest topological path.  It assumes every cell input reaches every
              output, so it counts the O5 -> I3 link along a whole ternary row, which is not a logic path.
  path_cells  the same longest path with the arcs each cell really has: a LUT output depends only on the pins its
              INIT reads, with constant pins fixed; CARRY4/CARRY8 output i depends on S[0..i], DI and CI.
  path_luts   that path counted in LUT and INV cells only (carry and MUXF cells weigh 0).
Cell counts are Yosys counts, not placed resources, and the paths are not timing.
"""
import argparse
import csv
import json
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
from functools import lru_cache
from pathlib import Path

import numpy as np

from da4ml.codegen.rtl.verilog import comb_logic_gen
from da4ml.trace import FixedVariableArray, HWConfig, comb_trace

FAMILIES = ('paper_8bit_positive', 'int4_signed_uniform')


def build_matrix(family, width, seed):
    rng = np.random.default_rng(seed)
    if family == 'paper_8bit_positive':
        return rng.integers(129, 256, size=(width, width), dtype=np.int64)
    return rng.integers(-8, 8, size=(width, width), dtype=np.int64)


def build_solution(matrix, dc):
    size = matrix.shape[0]
    x = FixedVariableArray.from_kif(
        np.ones(size, dtype=np.int64),
        np.full(size, 7, dtype=np.int64),
        np.zeros(size, dtype=np.int64),
        hwconf=HWConfig(1, -1, -1),
        solver_options={'hard_dc': dc},
    )
    return comb_trace(x, x @ matrix.T.astype(np.float32))


def _lut_arcs(cell):
    conn, typ = cell['connections'], cell['type']
    if typ == 'INV':
        return {'O': ['I']}
    init = cell['parameters']['INIT']
    init = int(init, 2) if set(init) <= {'0', '1'} else int(init)
    n = 6 if typ == 'LUT6_2' else int(typ[3])
    pins = [f'I{k}' for k in range(n)]
    const = {p: int(conn[p][0]) for p in pins if conn[p][0] in ('0', '1')}
    outs = {'O6': (init, 6), 'O5': (init & 0xFFFFFFFF, 5)} if typ == 'LUT6_2' else {'O': (init, n)}
    arcs = {}
    for o, (table, nb) in outs.items():
        used = pins[:nb]
        base = sum(1 << int(q[1]) for q, v in const.items() if int(q[1]) < nb and v)
        dep = []
        for p in used:
            if p in const:
                continue
            others = [int(q[1]) for q in used if q not in const and q != p]
            for a in range(1 << len(others)):
                idx = base | sum(1 << b for j, b in enumerate(others) if (a >> j) & 1)
                if ((table >> idx) & 1) != ((table >> (idx | 1 << int(p[1]))) & 1):
                    dep.append(p)
                    break
        arcs[o] = dep
    return arcs


def paths(module):
    """Longest output path with true cell arcs: (cells, LUT and INV cells)."""
    driver = {}
    for name, cell in module['cells'].items():
        t, c = cell['type'], cell['connections']
        if t.startswith('LUT') or t == 'INV':
            for o, dep in _lut_arcs(cell).items():
                for b in c[o]:
                    driver[b] = (1, [x for p in dep for x in c[p]])
        elif t in ('CARRY4', 'CARRY8'):
            extra = c['CYINIT'] if t == 'CARRY4' else []
            for port in ('O', 'CO'):
                for i, b in enumerate(c[port]):
                    dep = c['S'][: i + 1] + c['DI'][: i + (port == 'CO')] + c['CI'] + extra
                    if t == 'CARRY8' and i >= 4:
                        dep = dep + c.get('CI_TOP', [])
                    driver[b] = (0, dep)
        else:
            dirs = cell.get('port_directions', {})
            ins = [x for p, d in dirs.items() if d == 'input' for x in c[p]]
            for p, d in dirs.items():
                if d == 'output':
                    for b in c[p]:
                        driver[b] = (0, ins)

    @lru_cache(maxsize=None)
    def depth(b, luts_only):
        if not isinstance(b, int) or b not in driver:
            return 0
        w, dep = driver[b]
        return (w if luts_only else 1) + max([depth(x, luts_only) for x in dep] + [0])

    outs = [b for p in module['ports'].values() if p['direction'] == 'output' for b in p['bits']]
    return max(depth(b, False) for b in outs), max(depth(b, True) for b in outs)


def map_variant(yosys, source, rtl, workdir, name):
    inp = workdir / f'{name}.v'
    netlist = workdir / f'{name}.json'
    log = workdir / f'{name}.log'
    inp.write_text(rtl)
    cmd = (
        f'read_verilog {source / "shift_adder.v"} {source / "negative.v"} {inp}; '
        f'synth_xilinx -family xcup -nodsp -flatten -top {name}; '
        f'ltp -noff; write_json {netlist}'
    )
    p = subprocess.run([yosys, '-Q', '-T', '-p', cmd], capture_output=True, text=True, timeout=1800, check=False)
    log.write_text('$ yosys -Q -T -p ' + cmd + '\n' + p.stdout + p.stderr)
    if p.returncode:
        raise RuntimeError(f'{name} mapping failed: {log}')
    module = json.loads(netlist.read_text())['modules'][name]
    types = [c['type'] for c in module['cells'].values()]
    ltp = re.findall(r'Longest topological path in \S+ \(length=(\d+)\)', p.stdout)
    path_cells, path_luts = paths(module)
    return dict(
        lut=sum(bool(re.fullmatch(r'LUT[1-6](_2)?', t)) for t in types),
        inv=types.count('INV'),
        carry4=types.count('CARRY4'),
        carry8=types.count('CARRY8'),
        ltp=int(ltp[-1]) if ltp else None,
        path_cells=path_cells,
        path_luts=path_luts,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--count', type=int, default=5)
    parser.add_argument('--seed', type=int, default=20261121)
    parser.add_argument('--size', type=int, default=8)
    parser.add_argument('--dc', type=int, default=2, help='solver hard_dc (default 2, as the da4ml CLI)')
    parser.add_argument('--yosys', default=None, help='Yosys executable (default: yosys on PATH)')
    parser.add_argument('--out', type=Path, default=Path('ternary_results.csv'))
    args = parser.parse_args()
    yosys = args.yosys or shutil.which('yosys')
    if yosys is None:
        raise SystemExit('yosys executable required')
    version = subprocess.run([yosys, '-V'], capture_output=True, text=True).stdout.strip()
    print(version, flush=True)
    import da4ml

    source = Path(da4ml.__file__).resolve().parent / 'codegen/rtl/verilog/source'
    sys.setrecursionlimit(100000)
    rows = []
    with tempfile.TemporaryDirectory() as d:
        workdir = Path(d)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        for fi, family in enumerate(FAMILIES):
            for j in range(args.count):
                seed = args.seed + j + 50000 * fi
                sol = build_solution(build_matrix(family, args.size, seed), args.dc)
                row = dict(family=family, sample=j, seed=seed, size=args.size, hard_dc=args.dc)
                for key, tern in (('base', False), ('tern', True)):
                    mapped = map_variant(yosys, source, comb_logic_gen(sol, key, ternary_rows=tern), workdir, key)
                    row.update({f'{key}_{k}': v for k, v in mapped.items()})
                rows.append(row)
                print(json.dumps(row), flush=True)
    with args.out.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    keys = [k for k in rows[0] if k.startswith(('base_', 'tern_'))]
    for family in FAMILIES:
        subset = [r for r in rows if r['family'] == family]
        print(family, f'n={len(subset)}', f'hard_dc={args.dc}')
        for k in keys:
            vals = [r[k] for r in subset]
            print(f'  {k:16s} mean {statistics.mean(vals):9.2f}  min {min(vals):6d}  max {max(vals):6d}')


if __name__ == '__main__':
    main()
