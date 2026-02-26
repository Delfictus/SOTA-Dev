#!/usr/bin/env bash
set -euo pipefail

PRISM_ROOT="${PRISM_ROOT:-$HOME/Desktop/Prism4D-bio}"
NHS_BIN="$PRISM_ROOT/target/release/nhs_rt_full"
CCNS_BIN="$PRISM_ROOT/target/release/ccns-analyze"
PREP_DIR="$PRISM_ROOT/e2e_validation_test/prep"
RESULTS_DIR="/tmp/cryptosite_wave1_$(date +%Y%m%d_%H%M%S)"
RESULTS_JSON="$RESULTS_DIR/cryptosite_results.json"
LOG_FILE="$RESULTS_DIR/run.log"
LOCK_FILE="$RESULTS_DIR/results.lock"
mkdir -p "$RESULTS_DIR"
export RESULTS_JSON LOG_FILE LOCK_FILE NHS_BIN CCNS_BIN PREP_DIR RESULTS_DIR

echo "============================================================" | tee "$LOG_FILE"
echo " CryptoSite CCNS Benchmark — WAVE 1"                        | tee -a "$LOG_FILE"
echo " Config: sequential | 16 streams | no --multi-scale"        | tee -a "$LOG_FILE"
echo " Started: $(date)"                                          | tee -a "$LOG_FILE"
echo " Results: $RESULTS_DIR"                                     | tee -a "$LOG_FILE"
echo "============================================================" | tee -a "$LOG_FILE"

declare -A GROUND_TRUTH=(
    ["1a4q"]=1 ["1ade"]=1 ["1ere"]=1 ["1igt"]=1 ["4obe_mono"]=0
)

python3 -c "
import json, os
data = {'benchmark': 'CryptoSite_CCNS_Wave1', 'date': '$(date -Iseconds)', 'results': []}
json.dump(data, open(os.environ['RESULTS_JSON'], 'w'), indent=2)
"

run_protein() {
    local PDB="$1" LABEL="$2"
    local TOPOLOGY="$PREP_DIR/${PDB}.topology.json"
    local OUT_DIR="$RESULTS_DIR/$PDB"

    echo "" | tee -a "$LOG_FILE"
    echo "------------------------------------------------------------" | tee -a "$LOG_FILE"
    echo "  START: $PDB (ground_truth=$LABEL) — $(date +%H:%M:%S)"    | tee -a "$LOG_FILE"
    echo "------------------------------------------------------------" | tee -a "$LOG_FILE"

    write_error() {
        local STATUS="$1"
        flock "$LOCK_FILE" python3 -c "
import json, os
data = json.load(open(os.environ['RESULTS_JSON']))
data['results'].append({'pdb': '$PDB', 'ground_truth': $LABEL, 'status': '$STATUS',
    'tau': None, 'delta_g': None, 'classification': None, 'druggability_score': None})
json.dump(data, open(os.environ['RESULTS_JSON'], 'w'), indent=2)
"
    }

    if [ ! -f "$TOPOLOGY" ]; then
        echo "  SKIP $PDB: topology not found" | tee -a "$LOG_FILE"
        write_error "missing_topology"; return
    fi

    mkdir -p "$OUT_DIR"
    echo "  [$PDB] [1/2] nhs_rt_full..." | tee -a "$LOG_FILE"

    if RUST_LOG=info "$NHS_BIN" \
        -t "$TOPOLOGY" -o "$OUT_DIR" \
        --fast --multi-stream 8 --lining-cutoff 8.0 \
        --hysteresis --rt-clustering \
        2>&1 | tee -a "$OUT_DIR/nhs.log" "$LOG_FILE"; then
        echo "  [$PDB] NHS: OK — $(date +%H:%M:%S)" | tee -a "$LOG_FILE"
    else
        echo "  [$PDB] NHS FAILED" | tee -a "$LOG_FILE"
        write_error "nhs_failed"; return
    fi

    local SPIKE_FILE
    SPIKE_FILE=$(find "$OUT_DIR" -name "*.spike_events.json" 2>/dev/null | head -1)
    if [ -z "$SPIKE_FILE" ]; then
        echo "  [$PDB] No spike_events.json" | tee -a "$LOG_FILE"
        write_error "no_spikes"; return
    fi

    local CCNS_OUT="$OUT_DIR/${PDB}.ccns.json"
    echo "  [$PDB] [2/2] ccns-analyze..." | tee -a "$LOG_FILE"

    if RUST_LOG=info "$CCNS_BIN" --input "$SPIKE_FILE" --output "$CCNS_OUT" \
        2>&1 | tee -a "$OUT_DIR/ccns.log" "$LOG_FILE"; then
        echo "  [$PDB] CCNS: OK — $(date +%H:%M:%S)" | tee -a "$LOG_FILE"
    else
        echo "  [$PDB] CCNS FAILED" | tee -a "$LOG_FILE"
        write_error "ccns_failed"; return
    fi

    flock "$LOCK_FILE" python3 - "$SPIKE_FILE" "$CCNS_OUT" "$PDB" "$LABEL" <<'PYEOF' | tee -a "$LOG_FILE"
import json, sys, os
spike_file, ccns_out, pdb, label = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
try:
    ccns = json.load(open(ccns_out))
    site = (ccns.get('sites') or [{}])[0]
    tau = site.get('tau_exponent')
    cls = site.get('druggability_class', 'Unknown')
    dg  = (site.get('cft') or {}).get('delta_g_kcal_mol')
    score = max(0.05, min(0.95, 2.0 - tau)) if tau is not None else None
    data = json.load(open(os.environ['RESULTS_JSON']))
    data['results'].append({'pdb': pdb, 'ground_truth': label, 'status': 'ok',
        'tau': tau, 'delta_g': dg, 'classification': cls, 'druggability_score': score})
    json.dump(data, open(os.environ['RESULTS_JSON'], 'w'), indent=2)
    gt_str = "cryptic" if label else "control"
    correct = (tau is not None and tau < 1.5) == bool(label)
    print(f"\n  RESULT [{pdb}] [{'CORRECT' if correct else 'WRONG'}]: tau={tau:.4f} class={cls} dG={dg:.3f} score={score:.3f} GT={gt_str}")
except Exception as e:
    print(f"  PARSE ERROR [{pdb}]: {e}", file=sys.stderr)
PYEOF
}

echo "Launching sequential run..." | tee -a "$LOG_FILE"
for PDB in 1a4q 1ade 1ere 1igt 4obe_mono; do
    run_protein "$PDB" "${GROUND_TRUTH[$PDB]}"
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
print(f"\n{'='*65}\n WAVE 1 RESULTS\n{'='*65}")
print(f"{'PDB':<14} {'GT':>9} {'tau':>7} {'class':<16} {'dG':>10} {'score':>6}")
print("-"*65)
for r in sorted(results, key=lambda x: x['tau'] or 9):
    gt = "cryptic" if r['ground_truth'] else "control"
    pred = (r['tau'] or 9) < 1.5
    corr = pred == bool(r['ground_truth'])
    flag = "" if corr else " <<< WRONG"
    print(f"{r['pdb']:<14} {gt:>9} {r['tau']:>7.4f} {(r['classification'] or 'N/A'):<16} {(r['delta_g'] or 0):>10.3f} {(r['druggability_score'] or 0):>6.3f}{flag}")
if failed:
    print(f"\nFailed: {[r['pdb'] for r in failed]}")
if P == 0 or N == 0:
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
    pred=(r['tau'] or 9)<1.5; gt=bool(r['ground_truth'])
    if pred and gt: tp_t+=1
    elif pred and not gt: fp_t+=1
    elif not pred and gt: fn_t+=1
    else: tn_t+=1
sens=tp_t/P; spec=tn_t/N if N else 0; acc=(tp_t+tn_t)/len(results)
ppv=tp_t/(tp_t+fp_t) if (tp_t+fp_t) else 0
print(f"\n{'='*65}\n METRICS\n{'='*65}")
print(f"  N={len(results)} ({P} cryptic, {N} control)  TP={tp_t} FP={fp_t} TN={tn_t} FN={fn_t}")
print(f"  Sensitivity:{sens:.3f}  Specificity:{spec:.3f}  Precision:{ppv:.3f}  Accuracy:{acc:.3f}")
print(f"  AUC-ROC: {auc:.3f}")
verdict=("STRONG — publishable" if auc>=0.80 else "PROMISING — investor-ready" if auc>=0.70 else "MARGINAL — above chance" if auc>=0.60 else "BELOW THRESHOLD — revisit method")
print(f"\n  VERDICT: {verdict}")
data['summary']={'auc_roc':round(auc,4),'sensitivity':round(sens,3),'specificity':round(spec,3),'precision':round(ppv,3),'accuracy':round(acc,3),'tp':tp_t,'fp':fp_t,'tn':tn_t,'fn':fn_t,'verdict':verdict}
json.dump(data, open(rj,'w'), indent=2)
print(f"\n  Results: {rj}")
PYEOF

echo "" | tee -a "$LOG_FILE"
echo " Done: $(date)" | tee -a "$LOG_FILE"
echo "============================================================" | tee -a "$LOG_FILE"
