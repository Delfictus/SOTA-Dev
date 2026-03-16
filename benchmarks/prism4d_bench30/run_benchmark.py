#!/usr/bin/env python3
"""PRISM4D-BENCH30: Full benchmark with SR@1/SR@3/SR@N curve reporting."""
import json, subprocess, os, time, sys
import numpy as np

os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

m = json.load(open('benchmarks/prism4d_bench30/benchmark_manifest.json'))
gt = json.load(open('benchmarks/prism4d_bench30/ground_truth/ligand_centroids.json'))

print('PRISM4D-BENCH30: %d targets' % len(m['targets']))
print('=' * 80)

all_results = {}

for t in m['targets']:
    tid = str(t['id'])
    topo = 'benchmarks/prism4d_bench30/' + t['topology_file']
    outdir = 'benchmarks/prism4d_bench30/results/%s' % tid
    os.makedirs(outdir, exist_ok=True)

    apo = t['apo_pdb']
    holo = t['holo_pdb']
    lig = t.get('ligand_resname', '?')
    stype = t.get('site_type', '?')

    print('\n--- Target %s: %s -> %s (%s) [%s] ---' % (tid, apo, holo, lig, stype))
    sys.stdout.flush()

    start = time.time()
    r = subprocess.run([
        'target/release/nhs_rt_full', '-t', topo, '-o', outdir,
        '--fast', '--hysteresis', '--multi-stream', '8',
        '--spike-percentile', '95', '--prism-therm',
        '--hmr', '--adaptive-dt', '-v'
    ], capture_output=True, text=True, timeout=600)
    elapsed = time.time() - start

    bs_files = [f for f in os.listdir(outdir) if f.endswith('.binding_sites.json')]
    if not bs_files:
        print('  FAIL: no output (%ds)' % int(elapsed))
        if r.stderr:
            print('  %s' % r.stderr[-300:])
        sys.stdout.flush()
        continue

    with open(os.path.join(outdir, bs_files[0])) as f:
        data = json.load(f)
    sites = data.get('sites', [])

    if tid not in gt:
        print('  OK: %d sites, %ds - NO GROUND TRUTH' % (len(sites), int(elapsed)))
        sys.stdout.flush()
        continue

    lig_centroid = np.array(gt[tid]['centroid'])

    site_dists = []
    for s in sites:
        sc = np.array(s['centroid'])
        dcc = float(np.linalg.norm(sc - lig_centroid))
        site_dists.append(dcc)

    best = [999.0] * 11
    for n in range(1, 11):
        if len(site_dists) >= n:
            best[n] = min(site_dists[:n])

    all_results[tid] = {
        'apo': apo, 'holo': holo, 'lig': lig, 'type': stype,
        'n_sites': len(sites), 'elapsed': elapsed,
        'dcc_top1': best[1], 'dcc_top3': best[3],
        'dcc_top5': best[5], 'dcc_top10': best[10],
        'all_dcc': site_dists[:10],
        'best_dcc': min(site_dists) if site_dists else 999,
        'best_rank': site_dists.index(min(site_dists)) + 1 if site_dists else -1,
    }

    print('  %d sites, %ds | Top-1: %.1fA | Top-3: %.1fA | Top-5: %.1fA | Top-10: %.1fA | Best: %.1fA (rank %d)' % (
        len(sites), int(elapsed), best[1], best[3], best[5], best[10],
        min(site_dists) if site_dists else 999,
        site_dists.index(min(site_dists)) + 1 if site_dists else -1))

    for i, s in enumerate(sites[:5]):
        d = site_dists[i] if i < len(site_dists) else 999
        marker = ' ***' if d <= 4 else ' **' if d <= 5 else ' *' if d <= 8 else ''
        print('    #%d id=%s q=%.3f DCC=%.1f vol=%.0f%s' % (
            i + 1, s['id'], s.get('quality_score', 0), d, s.get('volume', 0), marker))
    sys.stdout.flush()

# ========== AGGREGATE REPORT ==========
print('\n' + '=' * 80)
print('PRISM4D-BENCH30 AGGREGATE RESULTS')
print('=' * 80)

n = len(all_results)
if n == 0:
    print('No results!')
    sys.exit(1)

thresholds = [4.0, 5.0, 8.0, 10.0]
print('\n%-20s' % 'Metric', end='')
for th in thresholds:
    print('  DCC<=%.0fA' % th, end='')
print()
print('-' * 60)

for topn_label, topn in [('SR@1', 1), ('SR@3', 3), ('SR@N+2', None), ('SR@5', 5), ('SR@10', 10)]:
    print('%-20s' % topn_label, end='')
    for th in thresholds:
        count = 0
        for tid, res in all_results.items():
            k = 3 if topn is None else topn
            dcc = 999
            if k <= len(res.get('all_dcc', [])):
                dcc = min(res['all_dcc'][:k])
            elif res.get('all_dcc'):
                dcc = min(res['all_dcc'])
            if dcc <= th:
                count += 1
        print('  %d/%d (%2.0f%%)' % (count, n, count / n * 100), end='')
    print()

print('\n%-20s' % 'Mean Top-1 DCC', end='')
vals = [r['dcc_top1'] for r in all_results.values() if r['dcc_top1'] < 900]
print('  %.1fA' % np.mean(vals) if vals else '  N/A')
print('%-20s' % 'Median Top-1 DCC', end='')
print('  %.1fA' % np.median(vals) if vals else '  N/A')
print('%-20s' % 'Mean Best DCC', end='')
vals = [r['best_dcc'] for r in all_results.values() if r['best_dcc'] < 900]
print('  %.1fA' % np.mean(vals) if vals else '  N/A')
print('%-20s' % 'Mean time/target', end='')
vals = [r['elapsed'] for r in all_results.values()]
print('  %.0fs' % np.mean(vals) if vals else '  N/A')

print('\n--- SR@N Curve (DCC <= 5A) ---')
print('N   SR@N   Cumulative targets')
for topn in range(1, 11):
    count = 0
    for res in all_results.values():
        if topn <= len(res.get('all_dcc', [])):
            if min(res['all_dcc'][:topn]) <= 5.0:
                count += 1
        elif res.get('all_dcc') and min(res['all_dcc']) <= 5.0:
            count += 1
    bar = '#' * count
    print('%2d  %2d/%d (%2.0f%%)  %s' % (topn, count, n, count / n * 100, bar))

print('\n--- By Pocket Type ---')
for ptype in ['orthosteric', 'cryptic', 'allosteric', 'PPI']:
    subset = {k: v for k, v in all_results.items() if v['type'] == ptype}
    if not subset:
        continue
    ns = len(subset)
    sr1_5 = sum(1 for v in subset.values() if v['dcc_top1'] <= 5.0)
    sr3_5 = sum(1 for v in subset.values() if v['dcc_top3'] <= 5.0)
    sr1_8 = sum(1 for v in subset.values() if v['dcc_top1'] <= 8.0)
    print('  %-12s  N=%2d  SR@1<=5A: %d/%d (%2.0f%%)  SR@3<=5A: %d/%d (%2.0f%%)  SR@1<=8A: %d/%d (%2.0f%%)' % (
        ptype, ns, sr1_5, ns, sr1_5 / ns * 100, sr3_5, ns, sr3_5 / ns * 100, sr1_8, ns, sr1_8 / ns * 100))

print('\n--- Per-Target Detail ---')
print('%4s %5s %5s %5s %-12s %6s %6s %6s %7s %5s' % (
    '#', 'APO', 'HOLO', 'LIG', 'TYPE', 'Top1', 'Top3', 'Top10', 'Best', 'Rank'))
print('-' * 75)
for tid in sorted(all_results.keys(), key=lambda x: int(x)):
    r = all_results[tid]
    m1 = '***' if r['dcc_top1'] <= 4 else '**' if r['dcc_top1'] <= 5 else '*' if r['dcc_top1'] <= 8 else ''
    print('%4s %5s %5s %5s %-12s %5.1fA %5.1fA %5.1fA %6.1fA %4d %s' % (
        tid, r['apo'], r['holo'], r['lig'], r['type'],
        r['dcc_top1'], r['dcc_top3'], r['dcc_top10'], r['best_dcc'], r['best_rank'], m1))

# Save JSON results
with open('benchmarks/prism4d_bench30/benchmark_results.json', 'w') as f:
    json.dump(all_results, f, indent=2, default=lambda x: float(x) if isinstance(x, np.floating) else x)

print('\n' + '=' * 80)
print('BENCHMARK COMPLETE - %d targets evaluated' % n)
print('Results saved to benchmarks/prism4d_bench30/benchmark_results.json')
print('=' * 80)
