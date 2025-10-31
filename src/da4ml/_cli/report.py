import argparse
import json
import re
from pathlib import Path


def parse_timing_summary(timing_summary: str):
    loc0 = timing_summary.find('Design Timing Summary')
    lines = timing_summary[loc0:].split('\n')[3:10]
    lines = [line for line in lines if line.strip() != '']

    assert set(lines[1]) == {' ', '-'}
    keys = [k.strip() for k in lines[0].split('  ') if k]
    vals = [int(v) if '.' not in v else float(v) for v in lines[2].split('  ') if v]
    assert len(keys) == len(vals)
    d = dict(zip(keys, vals))
    return d


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

mms = []
for name in track:
    m = re.compile(
        rf'\|\s*{name}\s*\|\s*(?P<Used>\d+)\s*\|\s*(?P<Fixed>\d+)\s*\|\s*(?P<Prohibited>\d+)\s*\|\s*(?P<Available>\d+)\s*\|'
    )
    mms.append(m)


def parse_utilization(utilization: str):
    """
    Parse the utilization report and return a DataFrame with the results.
    """

    dd = {}
    for name, m in zip(track, mms):
        found = m.findall(utilization)
        # assert found, f"{name} not found in utilization report"
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


def load_project(path: str | Path):
    path = Path(path)
    build_tcl_path = path / 'build_vivado_prj.tcl'
    assert build_tcl_path.exists(), f'build_prj.tcl not found in {path}'
    top_name = build_tcl_path.read_text().split('"', 2)[1]

    with open(path / f'src/{top_name}.xdc') as f:
        target_clock_period = float(f.readline().strip().split()[2])
    with open(path / 'metadata.json') as f:
        metadata = json.load(f)

    if metadata['flavor'] == 'vhdl':
        with open(path / f'src/{top_name}.vhd') as f:  # type: ignore
            latency = f.read().count('register') // 2
    else:
        with open(path / f'src/{top_name}.v') as f:  # type: ignore
            latency = f.read().count('reg') - 1

    d = {'clock_period': target_clock_period, 'latency': latency, **metadata}

    if (path / f'output_{top_name}/reports/{top_name}_post_route_util.rpt').exists():
        with open(path / f'output_{top_name}/reports/{top_name}_post_route_util.rpt') as f:
            util_rpt = f.read()
            util = parse_utilization(util_rpt)

        with open(path / f'output_{top_name}/reports/{top_name}_post_route_timing.rpt') as f:
            timing_rpt = f.read()
            timing = parse_timing_summary(timing_rpt)
        d.update(timing)
        d.update(util)

        d['actual_period'] = d['clock_period'] - d['WNS(ns)']

    return d


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


def report_main(args):
    vals = [load_project(Path(p)) for p in args.paths]
    for path, val in zip(args.paths, vals):
        d = extra_info_from_fname(Path(path).name)
        for k, v in d.items():
            val.setdefault(k, v)

    attrs: set[str] = set()
    for v in vals:
        attrs.update(v.keys())
    arr: list[list] = [list(attrs)]
    for v in vals:
        arr.append([v.get(a, '') for a in attrs])

    output = args.output
    if output == 'stdout':
        if not args.full:
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
            ]

            arr = [list(filter(lambda x: x in whitelist, arr[0]))]
            for row in vals:
                arr.append([row[a] for a in arr[0]])

        col_widths = [min(max(len(str(row[i])) for row in arr), 8) for i in range(len(arr[0]))]
        print('| ' + ' | '.join(f'{str(arr[0][i]).ljust(col_widths[i])[:8]}' for i in range(len(arr[0]))) + ' |')
        print('|-' + '-|-'.join('-' * col_widths[i] for i in range(len(arr[0]))) + '-|')
        for row in arr[1:]:
            print('| ' + ' | '.join(f'{str(row[i]).ljust(col_widths[i])[:8]}' for i in range(len(row))) + ' |')

        exit(0)

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
        '--full',
        '-f',
        action='store_true',
        help='Include full information for stdout output. For file output, all information will always be included.',
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Load HDL summaries')
    _add_report_args(parser)
    args = parser.parse_args()
    report_main(args)
