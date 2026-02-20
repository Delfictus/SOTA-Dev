#!/usr/bin/env bash
set -euo pipefail

PRISM_ROOT="${PRISM_ROOT:-$HOME/Desktop/Prism4D-bio}"
NHS_BIN="$PRISM_ROOT/target/release/nhs_rt_full"
CCNS_BIN="$PRISM_ROOT/target/release/ccns-analyze"
PREP_DIR="$PRISM_ROOT/e2e_validation_test/prep"
RESULTS_DIR="/tmp/cryptosite_bench_$(date +%Y%m%d_%H%M%S)"
RESULTS_JSON="$RESULTS_DIR/cryptosite_results.json"
LOG_FILE="$RESULTS_DIR/run.log"

mkdir -p "$RESULTS_DIR"
export RESULTS_JSON

echo "============================================================" | tee "$LOG_FILE"
echo " CryptoSite CCNS Benchmark" | tee -a "$LOG_FILE"
echo " Started: $(date)" | tee -a "$LOG_FILE"
echo " Results: $RESULTS_DIR" | tee -a "$LOG_FILE"
echo "============================================================" | tee -a "$LOG_FILE"

declare -A GROUND_TRUTH=(
    ["1a4q"]=1 ["1bj4"]=1 ["1dlo"]=1 ["1g1f"]=1
    ["1qmf"]=1 ["1w50"]=1 ["1hhp"]=1 ["2vsm"]=1
    ["1ere"]=1 ["1maz"]=1 ["1igt"]=1 ["3k5v"]=1
    ["1btl"]=1 ["1ade"]=1
    ["1nkp"]=0 ["1l2y"]=0 ["1crn"]=0 ["4obe_mono"]=0 ["1ubq"]=0
)

python3 -c "
import json, os
data = {'benchmark': 'CryptoSite_CCNS', 'date': '$(date -Iseconds)', 'results': []}
json.dump(data, open(os.environ['RESULTS_JSON'], 'w'), indent=2)
"

TOTAL=${#GROUND_TRUTH[@]}
DONE=0; FAILED=0; IDX=0

for PDB in "${!GROUND_TRUTH[@]}"; do
    IDX=$((IDX + 1))
    TOPOLOGY="$PREP_DIR/${PDB}.topology.json"
    LABEL="${GROUND_TRUTH[$PDB]}"

    echo "" | tee -a "$LOG_FILE"
    echo "------------------------------------------------------------" | tee -a "$LOG_FILE"
    echo "[$IDX/$TOTAL] $PDB (ground_truth=$LABEL) — $(date +%H:%M:%S)" | tee -a "$LOG_FILE"
    echo "------------------------------------------------------------" | tee -a "$LOG_FILE"

    if [ ! -f "$TOPOLOGY" ]; then
        echo "  SKIP: topology not found" | tee -a "$LOG_FILE"
        python3 -c "
import json, os
data = json.load(open(os.environ['RESULTS_JSON']))
data['results'].append({'pdb': '$PDB', 'ground_truth': $LABEL, 'status': 'missing_topology', 'tau': None, 'delta_g': None, 'classification': None, 'druggability_score': None})
json.dump(data, open(os.environ['RESULTS_JSON'], 'w'), indent=2)
"
        FAILED=$((FAILED + 1)); continue
    fi

    OUT_DIR="$RESULTS_DIR/$PDB"
    mkdir -p "$OUT_DIR"

    echo "  [1/2] nhs_rt_full --fast --multi-stream 8 --hysteresis --rt-clustering ..." | tee -a "$LOG_FILE"
    if RUST_LOG=warn "$NHS_BIN" \
        -t "$TOPOLOGY" \
        -o "$OUT_DIR" \
        --fast \
        --multi-stream 8 \
        --lining-cutoff 8.0 \
        --hysteresis \
        --rt-clustering \
        --multi-scale \
        2>> "$LOG_FILE"; then
        echo "  [1/2] NHS: OK" | tee -a "$LOG_FILE"
    else
        echo "  [1/2] NHS FAILED" | tee -a "$LOG_FILE"
        python3 -c "
import json, os
data = json.load(open(os.environ['RESULTS_JSON']))
data['results'].append({'pdb': '$PDB', 'ground_truth': $LABEL, 'status': 'nhs_failed', 'tau': None, 'delta_g': None, 'classification': None, 'druggability_score': None})
json.dump(data, open(os.environ['RESULTS_JSON'], 'w'), indent=2)
"
        FAILED=$((FAILED + 1)); continue
    fi

    SPIKE_FILE=$(find "$OUT_DIR" -name "*.spike_events.json" 2>/dev/null | head -1)
    if [ -z "$SPIKE_FILE" ]; then
        echo "  [1/2] No spike_events.json found" | tee -a "$LOG_FILE"
        python3 -c "
import json, os
data = json.load(open(os.environ['RESULTS_JSON']))
data['results'].append({'pdb': '$PDB', 'ground_truth': $LABEL, 'status': 'no_spikes', 'tau': None, 'delta_g': None, 'classification': None, 'druggability_score': None})
json.dump(data, open(os.environ['RESULTS_JSON'], 'w'), indent=2)
"
        FAILED=$((FAILED + 1)); continue
    fi

    CCNS_OUT="$OUT_DIR/${PDB}.ccns.json"
    echo "  [2/2] ccns-analyze ..." | tee -a "$LOG_FILE"
    if RUST_LOG=warn "$CCNS_BIN" \
        --input "$SPIKE_FILE" \
        --output "$CCNS_OUT" \
        2>> "$LOG_FILE"; then
        echo "  [2/2] CCNS: OK" | tee -a "$LOG_FILE"
    else
        echo "  [2/2] CCNS FAILED" | tee -a "$LOG_FILE"
        python3 -c "
import json, os
data = json.load(open(os.environ['RESULTS_JSON']))
data['results'].append({'pdb': '$PDB', 'ground_truth': $LABEL, 'status': 'ccns_failed', 'tau': None, 'delta_g': None, 'classification': None, 'druggability_score': None})
json.dump(data, open(os.environ['RESULTS_JSON'], 'w'), indent=2)
"
        FAILED=$((FAILED + 1)); continue
    fi

    python3 - "$SPIKE_FILE" "$CCNS_OUT" "$PDB" "$LABEL" << 'PYEOF'
import json, sys, os
spike_file, ccns_out, pdb, label = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
try:
    ccns = json.load(open(ccns_out))
    site = (ccns.get('sites') or [{}])[0]
    tau  = site.get('tau_exponent')
    cls  = site.get('druggability_class', 'Unknown')
    dg   = (site.get('cft') or {}).get('delta_g_kcal_mol')
    score = max(0.05, min(0.95, 2.0 - tau)) if tau is not None else None
    data = json.load(open(os.environ['RESULTS_JSON']))
    data['results'].append({
        'pdb': pdb, 'ground_truth': label, 'status': 'ok',
        'tau': tau, 'delta_g': dg, 'classification': cls, 'druggability_score': score,
    })
    json.dump(data, open(os.environ['RESULTS_JSON'], 'w'), indent=2)
    gt_str = "cryptic" if label else "control"
    correct = (tau is not None and tau < 1.5) == bool(label)
    print(f"  RESULT [{'CORRECT' if correct else 'WRONG'}]: tau={tau:.4f}, class={cls}, dG={dg:.3f}, score={score:.3f} (GT={gt_str})")
except Exception as e:
    print(f"  PARSE ERROR: {e}", file=sys.stderr)
    data = json.load(open(os.environ['RESULTS_JSON']))
    data['results'].append({'pdb': pdb, 'ground_truth': label, 'status': 'parse_error', 'tau': None, 'delta_g': None, 'classification': None, 'druggability_score': None})
    json.dump(data, open(os.environ['RESULTS_JSON'], 'w'), indent=2)
PYEOF

    DONE=$((DONE + 1))
    echo "  Progress: $DONE done, $FAILED failed" | tee -a "$LOG_FILE"
done

echo "" | tee -a "$LOG_FILE"
echo "============================================================" | tee -a "$LOG_FILE"
echo " FINAL ANALYSIS" | tee -a "$LOG_FILE"
echo "============================================================" | tee -a "$LOG_FILE"

python3 - << 'PYEOF'
import json, os, sys

rj = os.environ['RESULTS_JSON']
data = json.load(open(rj))
results = [r for r in data['results'] if r['status'] == 'ok' and r['tau'] is not None]
failed  = [r for r in data['results'] if r['status'] != 'ok']

P = sum(1 for r in results if r['ground_truth'])
N = len(results) - P

print(f"\n{'='*65}")
print(f" CRYPTOSITE CCNS BENCHMARK — RESULTS")
print(f"{'='*65}")
print(f"{'PDB':<14} {'GT':>9} {'tau':>7} {'class':<16} {'dG':>10} {'score':>6}")
print("-"*65)
for r in sorted(results, key=lambda x: x['tau'] or 9):
    gt = "cryptic" if r['ground_truth'] else "control"
    pred = (r['tau'] or 9) < 1.5
    corr = pred == bool(r['ground_truth'])
    flag = "" if corr else " <<< WRONG"
    print(f"{r['pdb']:<14} {gt:>9} {r['tau']:>7.4f} {(r['classification'] or 'N/A'):<16} {(r['delta_g'] or 0):>10.3f} {(r['druggability_score'] or 0):>6.3f}{flag}")

if failed:
    print(f"\nFailed ({len(failed)}): {[r['pdb'] for r in failed]}")

if len(results) < 4 or P == 0 or N == 0:
    print("\nInsufficient data for AUC."); sys.exit(0)

pairs = sorted(zip([r['druggability_score'] for r in results], [r['ground_truth'] for r in results]), key=lambda x: -x[0])
tp = fp = 0; fprs, tprs = [0.0], [0.0]
for score, label in pairs:
    if label: tp += 1
    else: fp += 1
    tprs.append(tp/P); fprs.append(fp/N)
auc = sum((fprs[i+1]-fprs[i])*(tprs[i+1]+tprs[i])/2 for i in range(len(fprs)-1))

tp_t=fp_t=tn_t=fn_t=0
for r in results:
    pred = (r['tau'] or 9) < 1.5
    gt = bool(r['ground_truth'])
    if pred and gt: tp_t+=1
    elif pred and not gt: fp_t+=1
    elif not pred and gt: fn_t+=1
    else: tn_t+=1

sens = tp_t/P; spec = tn_t/N if N else 0; acc = (tp_t+tn_t)/len(results)
ppv  = tp_t/(tp_t+fp_t) if (tp_t+fp_t) else 0

print(f"\n{'='*65}")
print(f" METRICS  (threshold: tau < 1.5 = cryptic)")
print(f"{'='*65}")
print(f"  N={len(results)} ({P} cryptic, {N} control)   TP={tp_t} FP={fp_t} TN={tn_t} FN={fn_t}")
print(f"  Sensitivity : {sens:.3f}   Specificity : {spec:.3f}")
print(f"  Precision   : {ppv:.3f}   Accuracy    : {acc:.3f}")
print(f"  AUC-ROC     : {auc:.3f}")

verdict = ("STRONG — publishable" if auc>=0.80 else
           "PROMISING — investor-ready" if auc>=0.70 else
           "MARGINAL — above chance" if auc>=0.60 else
           "BELOW THRESHOLD — revisit method")
print(f"\n  VERDICT: {verdict}")

data['summary'] = {'n_evaluated': len(results), 'n_failed': len(failed),
    'auc_roc': round(auc,4), 'sensitivity': round(sens,3), 'specificity': round(spec,3),
    'precision': round(ppv,3), 'accuracy': round(acc,3),
    'tp': tp_t, 'fp': fp_t, 'tn': tn_t, 'fn': fn_t, 'verdict': verdict}
json.dump(data, open(rj,'w'), indent=2)
print(f"\n  Results: {rj}")
PYEOF

echo "Finished: $(date)"
