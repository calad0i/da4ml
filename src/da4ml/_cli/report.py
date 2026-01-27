import argparse
import json
import os
import re
from collections.abc import Callable
from math import ceil, log10
from pathlib import Path
from typing import Any


def parse_timing_summary_vivado(timing_summary: str):
    loc0 = timing_summary.find('Design Timing Summary')
    lines = timing_summary[loc0:].split('\n')[3:10]
    lines = [line for line in lines if line.strip() != '']

    assert set(lines[1]) == {' ', '-'}
    keys = [k.strip() for k in lines[0].split('  ') if k]
    vals = [int(v) if '.' not in v else float(v) for v in lines[2].split('  ') if v]
    assert len(keys) == len(vals)
    d = dict(zip(keys, vals))
    return d


def parse_utilization_vivado(utilization: str):
    """
    Parse the utilization report and return a DataFrame with the results.
    """
    track = [
        'DSPs',
        'LUT as Logic',
        'LUT as Memory',
        'CLB Registers',
        'CARRY8',
        'Register as Latch',
        'Register as Flip Flop',
        'RAMB18',
        'URAM',
        'RAMB36/FIFO*',
    ]
    matchers = []
    for name in track:
        m = re.compile(
            rf'\|\s*{name}\s*\|\s*(?P<Used>\d+)\s*\|\s*(?P<Fixed>\d+)\s*\|\s*(?P<Prohibited>\d+)\s*\|\s*(?P<Available>\d+)\s*\|'
        )
        matchers.append(m)

    dd = {}
    for name, m in zip(track, matchers):
        found = m.findall(utilization)
        used, fixed, prohibited, available = map(int, found[0])
        dd[name] = used
        dd[f'{name}_fixed'] = fixed
        dd[f'{name}_prohibited'] = prohibited
        dd[f'{name}_available'] = available

    dd['FF'] = dd['Register as Flip Flop'] + dd['Register as Latch']
    dd['LUT'] = dd['LUT as Logic'] + dd['LUT as Memory']
    dd['LUT_available'] = max(dd['LUT as Logic_available'], dd['LUT as Memory_available'])
    dd['FF_available'] = max(dd['Register as Flip Flop_available'], dd['Register as Latch_available'])
    dd['DSP'] = dd['DSPs']

    return dd


def parse_power_vivado(power_report: str):
    matchers = []
    track = ['Total On-Chip Power (W)', 'Dynamic (W)', 'Device Static (W)']
    for name in track:
        m = re.compile(rf'\|\s*{re.escape(name)}\s*\|\s*(?P<Value>[^\|]+?)\s*\|')
        matchers.append(m)
    dd = {}
    for name, m in zip(track, matchers):
        found = m.findall(power_report)
        value = found[0].strip()
        dd[name] = value
    return dd


def parse_timing_quartus(sta_report: str) -> dict[str, Any]:
    """
    Parse Altera/Quartus timing report (model.sta.rpt) for timing information.
    """
    dd: dict[str, Any] = {}

    fmax_m = re.compile(r';\s*(?P<Fmax>[\d.]+)\s*MHz\s*;\s*(?P<RestrictedFmax>[\d.]+)\s*MHz\s*;\s*(?P<ClockName>[^;]+?)\s*;')
    match = fmax_m.search(sta_report)
    if match:
        dd['Fmax(MHz)'] = float(match.group('Fmax'))
        dd['Restricted Fmax(MHz)'] = float(match.group('RestrictedFmax'))

    setup_blk_m = re.search(r'; Setup Summary\s*;.*?\n\+[-+]+\+\n(.*?)\n\+[-+]+\+', sta_report, re.DOTALL)
    if setup_blk_m:
        setup_txt = setup_blk_m.group(1)
        setup_data = re.search(
            r';\s*(?P<Clock>[^;]+?)\s*;\s*(?P<Slack>-?[\d.]+)\s*;\s*(?P<TNS>-?[\d.]+)\s*;\s*(?P<FailingEndpoints>\d+)\s*;',
            setup_txt,
        )
        if setup_data:
            dd['Setup Slack'] = float(setup_data.group('Slack'))
            dd['Setup TNS'] = float(setup_data.group('TNS'))
            dd['Setup Failing Endpoints'] = int(setup_data.group('FailingEndpoints'))

    hold_section = re.search(r'; Hold Summary\s*;.*?\n\+[-+]+\+\n(.*?)\n\+[-+]+\+', sta_report, re.DOTALL)
    if hold_section:
        hold_data = re.search(
            r';\s*(?P<Clock>[^;]+?)\s*;\s*(?P<Slack>-?[\d.]+)\s*;\s*(?P<TNS>-?[\d.]+)\s*;\s*(?P<FailingEndpoints>\d+)\s*;',
            hold_section.group(1),
        )
        if hold_data:
            dd['Hold Slack'] = float(hold_data.group('Slack'))
            dd['Hold TNS'] = float(hold_data.group('TNS'))
            dd['Hold Failing Endpoints'] = int(hold_data.group('FailingEndpoints'))

    return dd


def parse_utilization_quartus(fit_report: str) -> dict[str, Any]:
    """
    Parse Altera/Quartus fitter report (model.fit.rpt) for resource utilization.
    """
    dd: dict[str, Any] = {}

    summary_patterns = [
        (r';\s*Logic utilization \(in ALMs\)\s*;\s*(?P<Used>[\d,]+)\s*/\s*(?P<Available>[\d,]+)', 'ALMs'),
        (r';\s*Total dedicated logic registers\s*;\s*(?P<Used>[\d,]+)', 'Registers'),
        (r';\s*Total block memory bits\s*;\s*(?P<Used>[\d,]+)\s*/\s*(?P<Available>[\d,]+)', 'Block Memory Bits'),
        (r';\s*Total RAM Blocks\s*;\s*(?P<Used>[\d,]+)\s*/\s*(?P<Available>[\d,]+)', 'RAM Blocks'),
        (r';\s*Total DSP Blocks\s*;\s*(?P<Used>[\d,]+)\s*/\s*(?P<Available>[\d,]+)', 'DSP'),
        (r';\s*Total PLLs\s*;\s*(?P<Used>[\d,]+)\s*/\s*(?P<Available>[\d,]+)', 'PLLs'),
    ]

    for pattern, name in summary_patterns:
        match = re.search(pattern, fit_report)
        if not match:
            continue
        used = int(match.group('Used').replace(',', ''))
        dd[name] = used
        if 'Available' in match.groupdict():
            available = int(match.group('Available').replace(',', ''))
            dd[f'{name}_available'] = available

    detail_patterns = [
        (r';\s*Combinational ALUT usage for logic\s*;\s*(?P<Value>[\d,]+)', 'Combinational ALUTs'),
        (r';\s*Dedicated logic registers\s*;\s*(?P<Value>[\d,]+)', 'FF'),
        (r';\s*M20K blocks\s*;\s*(?P<Value>[\d,]+)\s*/', 'M20K'),
        (r';\s*Total MLAB memory bits\s*;\s*(?P<Value>[\d,]+)', 'MLAB Bits'),
        (r';\s*DSP Blocks Needed \[=A\+B\+C-D\]\s*;\s*(?P<Value>[\d,]+)\s*/', 'DSP'),
    ]

    for pattern, name in detail_patterns:
        match = re.search(pattern, fit_report)
        if match:
            value = int(match.group('Value').replace(',', ''))
            dd[name] = value

    # Map to common names
    if 'Combinational ALUTs' in dd:
        dd['LUT'] = dd['Combinational ALUTs']
    if 'ALMs' in dd:
        dd['ALM'] = dd['ALMs']

    return dd


def parse_if_exists(path: Path, parser_func: Callable[[str], dict[str, Any]]) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with open(path) as f:
        content = f.read()
    return parser_func(content)


def _load_project(path: str | Path) -> dict[str, Any]:
    """
    Load project summary from the given path.
    """
    path = Path(path)
    build_tcl_path = path / 'build_vivado_prj.tcl' if (path / 'build_vivado_prj.tcl').exists() else path / 'build_quartus_prj.tcl'
    assert build_tcl_path.exists(), f'build_vivado_prj.tcl or build_quartus_prj.tcl not found in {path}'
    top_name = build_tcl_path.read_text().split('"', 2)[1]
    rpt_path = path / f'output_{top_name}/reports/'

    # Read clock period from constraints file
    xdc_path = path / f'src/{top_name}.xdc'
    with open(xdc_path) as f:
        target_clock_period = float(f.readline().strip().split()[2])

    with open(path / 'metadata.json') as f:
        metadata = json.load(f)

    suffix = 'vhd' if metadata['flavor'] == 'vhdl' else 'v'
    with open(path / f'src/{top_name}.{suffix}') as f:
        latency = f.read().count('<=') - 1

    d = {**metadata, 'clock_period': target_clock_period, 'latency': latency}

    # Vivado reports
    util = parse_if_exists(rpt_path / f'{top_name}_post_route_util.rpt', parse_utilization_vivado)
    timing = parse_if_exists(rpt_path / f'{top_name}_post_route_timing.rpt', parse_timing_summary_vivado)
    power = parse_if_exists(rpt_path / f'{top_name}_post_route_power.rpt', parse_power_vivado)
    if util is not None:
        d.update(util)
    if timing is not None:
        d.update(timing)
        d['actual_period'] = d['clock_period'] - d['WNS(ns)']
        d['Fmax(MHz)'] = 1000.0 / d['actual_period']
        d['latency(ns)'] = d['latency'] * d['actual_period']
    if power is not None:
        d.update(power)

    # Quartus reports
    util = parse_if_exists(rpt_path / f'{top_name}.fit.rpt', parse_utilization_quartus)
    timing = parse_if_exists(rpt_path / f'{top_name}.sta.rpt', parse_timing_quartus)
    if timing is not None:
        d.update(timing)
        if 'Fmax(MHz)' in timing:
            d['actual_period'] = 1000.0 / timing['Fmax(MHz)']
            d['latency(ns)'] = d['latency'] * d['actual_period']
    if util is not None:
        d.update(util)

    return d


def load_project(path: str | Path) -> dict[str, Any] | None:
    try:
        return _load_project(path)
    except Exception as e:
        print(e)
        return None


def extra_info_from_fname(fname: str):
    d = {}
    for part in fname.split('-'):
        if '=' not in part:
            continue
        k, v = part.split('=', 1)
        try:
            v = int(v)
            d[k] = v
            continue
        except ValueError:
            pass
        try:
            v = float(v)
            d[k] = v
            continue
        except ValueError:
            pass
        d[k] = v
    return d


def pretty_print(arr: list[list]):
    n_cols = len(arr[0])
    terminal_width = os.get_terminal_size().columns
    default_width = [
        max(min(6, len(str(arr[i][j]))) if isinstance(arr[i][j], float) else len(str(arr[i][j])) for i in range(len(arr)))
        for j in range(n_cols)
    ]
    if sum(default_width) + 2 * n_cols + 1 <= terminal_width:
        col_width = default_width
    else:
        th = max(8, (terminal_width - 2 * n_cols - 1) // n_cols)
        col_width = [min(w, th) for w in default_width]

    header = [
        '| ' + ' | '.join(f'{str(arr[0][i]).ljust(col_width[i])[: col_width[i]]}' for i in range(n_cols)) + ' |',
        '|-' + '-|-'.join('-' * col_width[i] for i in range(n_cols)) + '-|',
    ]
    content = []
    for row in arr[1:]:
        _row = []
        for i, v in enumerate(row):
            w = col_width[i]
            if type(v) is float:
                n_int = ceil(log10(abs(v) + 1)) if v != 0 else 1 + (v < 0)
                v = round(v, 10 - n_int)
                if type(v) is int:
                    fmt = f'{{:>{w}d}}'
                    _v = fmt.format(v)
                else:
                    _v = str(v)
                    if len(_v) > w:
                        fmt = f'{{:.{max(w - n_int - 1, 0)}f}}'
                        _v = fmt.format(v).ljust(w)
                    else:
                        _v = _v.ljust(w)
            else:
                _v = str(v).ljust(w)[:w]
            _row.append(_v)
        content.append('| ' + ' | '.join(_row) + ' |')
    print('\n'.join(header + content))


def stdout_print(arr: list[list], full: bool, columns: list[str] | None):
    whitelist = [
        'epoch',
        'flavor',
        'actual_period',
        'clock_period',
        'ebops',
        'cost',
        'latency',
        'DSP',
        'LUT',
        'FF',
        'comb_metric',
        'Fmax(MHz)',
        'latency(ns)',
    ]
    if columns is None:
        columns = whitelist

    if not full:
        idx_row = arr[0]
        keep_cols = [idx_row.index(col) for col in columns if col in idx_row]
        arr = [[row[i] for i in keep_cols] for row in arr]

    if len(arr) == 2:  # One sample
        k_width = max(len(str(h)) for h in arr[0])
        for k, v in zip(arr[0], arr[1]):
            print(f'{str(k).ljust(k_width)} : {v}')
    else:
        pretty_print(arr)


def report_main(args):
    _vals = [load_project(Path(p)) for p in args.paths]
    vals = [v for v in _vals if v is not None]
    for path, val in zip(args.paths, vals):
        d = extra_info_from_fname(Path(path).name)
        for k, v in d.items():
            val.setdefault(k, v)

    _key = [x.get(args.sort_by, float('inf')) for x in vals]
    _order = sorted(range(len(vals)), key=lambda i: -_key[i])
    vals = [vals[i] for i in _order]

    _attrs: set[str] = set()
    for v in vals:
        _attrs.update(v.keys())
    attrs = sorted(_attrs)
    arr: list[list] = [attrs]
    for v in vals:
        arr.append([v.get(a, '') for a in attrs])

    output = args.output
    if output == 'stdout':
        stdout_print(arr, args.full, args.columns)
        return

    with open(output, 'w') as f:
        ext = Path(output).suffix
        if ext == '.json':
            json.dump(vals, f)
        elif ext in ['.tsv', '.csv']:
            sep = ',' if ext == '.csv' else '\t'
            op = (lambda x: str(x) if ',' not in str(x) else f'"{str(x)}"') if ext == '.csv' else lambda x: str(x)
            for row in arr:
                f.write(sep.join(map(op, row)) + '\n')  # type: ignore
        elif ext == '.md':
            f.write('| ' + ' | '.join(map(str, arr[0])) + ' |\n')
            f.write('|' + '|'.join(['---'] * len(arr[0])) + '|\n')
            for row in arr[1:]:
                f.write('| ' + ' | '.join(map(str, row)) + ' |\n')
        elif ext == '.html':
            f.write('<table>\n')
            f.write('  <tr>' + ''.join([f'<th>{a}</th>' for a in arr[0]]) + '</tr>\n')
            for row in arr[1:]:
                f.write('  <tr>' + ''.join([f'<td>{a}</td>' for a in row]) + '</tr>\n')
            f.write('</table>\n')
        else:
            raise ValueError(f'Unsupported output format: {ext}')


def _add_report_args(parser: argparse.ArgumentParser):
    parser.add_argument('paths', type=str, nargs='+', help='Paths to the directories containing HDL summaries')
    parser.add_argument(
        '--output',
        '-o',
        type=str,
        default='stdout',
        help='Output file name for the summary. Can be stdout, .json, .csv, .tsv, .md, .html',
    )
    parser.add_argument(
        '--sort-by',
        '-s',
        type=str,
        default='comb_metric',
        help='Attribute to sort the summary by. Default is cost.',
    )
    parser.add_argument(
        '--full',
        '-f',
        action='store_true',
        help='Include full information for stdout output. For file output, all information will always be included.',
    )
    parser.add_argument(
        '--columns',
        '-c',
        type=str,
        nargs='+',
        default=None,
        help='Specify columns to include in the report. Only applicable for stdout output. Ignored if --full is set.',
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Load HDL summaries')
    _add_report_args(parser)
    args = parser.parse_args()
    report_main(args)
