"""Summarize benchmark CSVs (Python V0 or C++ variants, same schema) into one table.

For every (device, variant, precision): median ms per stage, e2e median / p90, FPS and speedup
against a baseline run. Optionally writes the table as benchmark_results.csv.

Examples:
    python benchmarks/summarize.py benchmarks/results/*.csv
    python benchmarks/summarize.py benchmarks/results/*.csv --baseline V0 --out benchmark_results.csv
"""
import argparse, csv
from collections import defaultdict
import numpy as np

STAGES = ['preprocess', 'voxelize', 'h2d', 'pillar_enc', 'backbone', 'ego_motion', 'motion_seg', 'cluster']
EXTRA = ['tubenet']


def load(paths):
    data = defaultdict(lambda: defaultdict(list))     # (device, variant, precision) -> stage -> [ms]
    frames = defaultdict(set)
    for p in paths:
        with open(p, newline='') as fh:
            for r in csv.DictReader(fh):
                key = (r['device'], r['variant'], r['precision'])
                data[key][r['stage']].append(float(r['ms']))
                frames[key].add(r['frame'])
    return data, frames


def summarize(paths, baseline='V0', baseline_device=None, out=None):
    data, frames = load(paths)
    if not data:
        print('no rows')
        return []
    base = [k for k in data if k[1] == baseline and (baseline_device is None or k[0] == baseline_device)]
    base_e2e = np.median(data[base[0]]['e2e']) if base else None

    table = []
    for key in sorted(data, key=lambda k: (k[0], k[1], k[2])):
        st = data[key]
        row = {'device': key[0], 'variant': key[1], 'precision': key[2], 'frames': len(frames[key])}
        for s in STAGES + EXTRA:
            row[f'{s}_ms'] = round(float(np.median(st[s])), 2) if st.get(s) else ''
        e2e = np.asarray(st['e2e'])
        row['e2e_ms'] = round(float(np.median(e2e)), 2)
        row['e2e_p90_ms'] = round(float(np.percentile(e2e, 90)), 2)
        row['fps'] = round(1000.0 / float(e2e.mean()), 2)
        if st.get('e2e_nosync'):
            nos = np.asarray(st['e2e_nosync'])
            row['e2e_nosync_ms'] = round(float(np.median(nos)), 2)
            row['fps_nosync'] = round(1000.0 / float(nos.mean()), 2)
        else:
            row['e2e_nosync_ms'] = row['fps_nosync'] = ''
        row['speedup'] = round(float(base_e2e / np.median(e2e)), 2) if base_e2e else ''
        table.append(row)

    # console table: stages as rows, runs as columns
    cols = [f"{r['variant']} {r['precision']}" for r in table]
    devs = [r['device'].replace('NVIDIA ', '').replace('GeForce ', '') for r in table]
    w = max(14, *(len(c) + 2 for c in cols + devs))
    print('\n' + ' ' * 16 + ''.join(f'{d:>{w}}' for d in devs))
    print(' ' * 16 + ''.join(f'{c:>{w}}' for c in cols))
    lines = [(f'{s} ms', f'{s}_ms') for s in STAGES] + [('-' * 14, None), ('e2e ms', 'e2e_ms'),
             ('e2e p90 ms', 'e2e_p90_ms'), ('FPS', 'fps'), ('e2e nosync ms', 'e2e_nosync_ms'),
             ('FPS nosync', 'fps_nosync'), (f'speedup vs {baseline}', 'speedup'), ('tubenet ms*', 'tubenet_ms')]
    for label, k in lines:
        if k is None:
            print(label)
            continue
        print(f'{label:16s}' + ''.join(f'{str(r[k]):>{w}}' for r in table))
    print('(medians over all frames and passes; FPS = 1000 / mean e2e; e2e = preprocess..cluster;'
          ' *tubenet is outside the C++ scope)')
    if base_e2e is None:
        print(f'(no {baseline} run found: speedup left empty)')

    if out:
        with open(out, 'w', newline='') as fh:
            wr = csv.DictWriter(fh, fieldnames=list(table[0].keys()))
            wr.writeheader()
            wr.writerows(table)
        print(f'wrote {out}')
    return table


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('csv', nargs='+')
    ap.add_argument('--baseline', default='V0', help='variant used for the speedup column')
    ap.add_argument('--baseline-device', help='device of the baseline run (default: first match)')
    ap.add_argument('--out', help='write the summary table to this CSV (e.g. benchmark_results.csv)')
    a = ap.parse_args()
    summarize(a.csv, a.baseline, a.baseline_device, a.out)
