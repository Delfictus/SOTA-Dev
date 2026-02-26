#!/usr/bin/env bash
set -euo pipefail

if [ $# -lt 2 ]; then
    echo "Usage: $0 <batch_number> <target1> [target2] ..."
    echo "Example: $0 02 5caz 5tvi 2iyt 8j11 5yhb 4p2f 3w90 2fhz 1vsn 1zm0"
    exit 1
fi

BATCH="$1"; shift
TARGETS=("$@")

RESULTS_DIR="benchmarks/cryptobench/results"
TOPO_DIR="benchmarks/cryptobench/topologies"
OUTPUT_ROOT="benchmarks/cryptobench/aggregated/batch_${BATCH}"
TIMESTAMP=$(date -u +%Y%m%dT%H%M%SZ)

echo "╔═══════════════════════════════════════════════════════════════╗"
echo "║  PRISM-4D Batch ${BATCH} — Results Aggregator                    ║"
echo "║  Targets: ${#TARGETS[@]}                                               ║"
echo "║  Timestamp: ${TIMESTAMP}                                ║"
echo "╚═══════════════════════════════════════════════════════════════╝"
echo ""

mkdir -p "${OUTPUT_ROOT}"
for TGT in "${TARGETS[@]}"; do
    mkdir -p "${OUTPUT_ROOT}/${TGT}"/{json,spike_events,trajectories,visualization,reports}
done
mkdir -p "${OUTPUT_ROOT}/_merged"

echo "--- Collecting per-target files ---"
echo ""

TOTAL_FILES=0
for TGT in "${TARGETS[@]}"; do
    SRC="${RESULTS_DIR}/${TGT}"
    DST="${OUTPUT_ROOT}/${TGT}"
    FILE_COUNT=0

    if [ ! -d "${SRC}" ]; then
        echo "  ⚠  ${TGT}: results directory not found — SKIPPING"
        continue
    fi

    [ -f "${SRC}/${TGT}.binding_sites.json" ] && \
        cp "${SRC}/${TGT}.binding_sites.json" "${DST}/json/" && ((FILE_COUNT++)) || \
        echo "  ⚠  ${TGT}: binding_sites.json MISSING"

    for f in "${SRC}"/*.sndc_sites.json; do
        [ -f "$f" ] && cp "$f" "${DST}/json/" && ((FILE_COUNT++))
    done

    SPIKE_COUNT=0
    for f in "${SRC}"/${TGT}.site*.spike_events.json; do
        [ -f "$f" ] && cp "$f" "${DST}/spike_events/" && ((SPIKE_COUNT++)) && ((FILE_COUNT++))
    done

    TRAJ_COUNT=0
    for f in "${SRC}"/${TGT}_stream*.ensemble_trajectory.pdb; do
        [ -f "$f" ] && cp "$f" "${DST}/trajectories/" && ((TRAJ_COUNT++)) && ((FILE_COUNT++))
    done

    for ext in pml cxc; do
        [ -f "${SRC}/${TGT}.binding_sites.${ext}" ] && \
            cp "${SRC}/${TGT}.binding_sites.${ext}" "${DST}/visualization/" && ((FILE_COUNT++))
    done
    [ -f "${SRC}/${TGT}.binding_sites.pdb" ] && \
        cp "${SRC}/${TGT}.binding_sites.pdb" "${DST}/visualization/" && ((FILE_COUNT++))

    [ -f "${SRC}/${TGT}.binding_sites.md" ] && \
        cp "${SRC}/${TGT}.binding_sites.md" "${DST}/reports/" && ((FILE_COUNT++))

    [ -f "${TOPO_DIR}/${TGT}.topology.json" ] && \
        cp "${TOPO_DIR}/${TGT}.topology.json" "${DST}/json/" && ((FILE_COUNT++))

    for f in "${SRC}"/${TGT}.*; do
        BASENAME=$(basename "$f")
        case "${BASENAME}" in
            *.binding_sites.json|*.binding_sites.pdb|*.binding_sites.pml|\
            *.binding_sites.cxc|*.binding_sites.md|*.spike_events.json|\
            *.sndc_sites.json) continue ;;
        esac
        [ -f "$f" ] && cp "$f" "${DST}/json/" && ((FILE_COUNT++)) && \
            echo "  +  ${TGT}: extra file → ${BASENAME}"
    done

    TOTAL_FILES=$((TOTAL_FILES + FILE_COUNT))
    printf "  ✓ %-6s %3d files  (%d spike_events, %d trajectories)\n" "${TGT}:" "${FILE_COUNT}" "${SPIKE_COUNT}" "${TRAJ_COUNT}"
done

echo ""
echo "--- Building merged outputs ---"

MERGED="${OUTPUT_ROOT}/_merged/batch_${BATCH}_all_binding_sites.json"
echo "{" > "${MERGED}"
FIRST=true
for TGT in "${TARGETS[@]}"; do
    BS="${OUTPUT_ROOT}/${TGT}/json/${TGT}.binding_sites.json"
    [ ! -f "${BS}" ] && continue
    [ "${FIRST}" = true ] && FIRST=false || echo "," >> "${MERGED}"
    printf '  "%s": ' "${TGT}" >> "${MERGED}"
    cat "${BS}" >> "${MERGED}"
done
echo "" >> "${MERGED}"
echo "}" >> "${MERGED}"
echo "  ✓ ${MERGED}"

MANIFEST="${OUTPUT_ROOT}/_merged/manifest.tsv"
printf "target\tatoms\tnum_sites\tnum_spike_jsons\tnum_streams\n" > "${MANIFEST}"
for TGT in "${TARGETS[@]}"; do
    DST="${OUTPUT_ROOT}/${TGT}"
    N_ATOMS="?"; N_SITES="?"
    BS="${DST}/json/${TGT}.binding_sites.json"
    TOPO="${DST}/json/${TGT}.topology.json"
    [ -f "${BS}" ] && N_SITES=$(python3 -c "
import json
with open('${BS}') as f: d=json.load(f)
if isinstance(d,list): print(len(d))
elif 'sites' in d: print(len(d['sites']))
elif 'binding_sites' in d: print(len(d['binding_sites']))
else: print('?')
" 2>/dev/null || echo "?")
    [ -f "${TOPO}" ] && N_ATOMS=$(python3 -c "
import json
with open('${TOPO}') as f: d=json.load(f)
print(d.get('num_atoms',d.get('n_atoms','?')))
" 2>/dev/null || echo "?")
    N_SPIKE=$(find "${DST}/spike_events/" -name "*.spike_events.json" 2>/dev/null | wc -l)
    N_TRAJ=$(find "${DST}/trajectories/" -name "*.ensemble_trajectory.pdb" 2>/dev/null | wc -l)
    printf "%s\t%s\t%s\t%s\t%s\n" "${TGT}" "${N_ATOMS}" "${N_SITES}" "${N_SPIKE}" "${N_TRAJ}" >> "${MANIFEST}"
done
echo "  ✓ ${MANIFEST}"

SPIKE_IDX="${OUTPUT_ROOT}/_merged/spike_events_index.json"
echo "{" > "${SPIKE_IDX}"
FIRST=true
for TGT in "${TARGETS[@]}"; do
    FILES=$(find "${OUTPUT_ROOT}/${TGT}/spike_events" -name "*.json" -printf '%f\n' 2>/dev/null | sort)
    [ -z "${FILES}" ] && continue
    [ "${FIRST}" = true ] && FIRST=false || echo "," >> "${SPIKE_IDX}"
    printf '  "%s": [' "${TGT}" >> "${SPIKE_IDX}"
    INNER=true
    while IFS= read -r fn; do
        [ "${INNER}" = true ] && INNER=false || printf "," >> "${SPIKE_IDX}"
        printf '"%s"' "${fn}" >> "${SPIKE_IDX}"
    done <<< "${FILES}"
    printf "]" >> "${SPIKE_IDX}"
done
echo "" >> "${SPIKE_IDX}"
echo "}" >> "${SPIKE_IDX}"
echo "  ✓ ${SPIKE_IDX}"

echo ""
echo "╔═══════════════════════════════════════════════════════════════╗"
printf "║  BATCH %s COMPLETE — %d files across %d targets              ║\n" "${BATCH}" "${TOTAL_FILES}" "${#TARGETS[@]}"
echo "║  Output: ${OUTPUT_ROOT}"
echo "╚═══════════════════════════════════════════════════════════════╝"
