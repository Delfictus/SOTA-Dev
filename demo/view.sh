#!/bin/bash
CYAN='\033[38;2;0;229;255m'
WHITE='\033[1;38;2;232;236;240m'
DIM='\033[38;2;85;96;112m'
GREEN='\033[32;1m'
RED='\033[31;1m'
RESET='\033[0m'
OUTPUT_BASE="/opt/prism4d/output"

if [ -n "$1" ] && [ -d "$1" ]; then TARGET="$1"
elif [ -n "$1" ] && [ -d "$OUTPUT_BASE/$1" ]; then TARGET="$OUTPUT_BASE/$1"
else TARGET=$(ls -dt "$OUTPUT_BASE"/*/ 2>/dev/null | head -1); fi

if [ -z "$TARGET" ] || [ ! -d "$TARGET" ]; then
    echo -e "  ${RED}No output directory found${RESET}"
    echo -e "  ${DIM}Run a PRISM-4D analysis first, then use: view${RESET}"; exit 1
fi

DIRNAME=$(basename "$TARGET")
PDB_ID=$(echo "$DIRNAME" | cut -d'_' -f1)

echo ""
echo -e "  ${CYAN}━━━ PRISM-4D Viewer Generator ━━━${RESET}"
echo -e "  ${DIM}Run: ${DIRNAME}${RESET}"
echo ""

PML=$(find "$TARGET" -name "*.binding_sites.pml" 2>/dev/null | head -1)
if [ -z "$PML" ]; then
    echo -e "  ${RED}No .binding_sites.pml found in ${TARGET}${RESET}"; exit 1
fi

python3 /opt/prism4d/scripts/generate_viewer.py "$TARGET"
RET=$?

if [ $RET -eq 0 ]; then
    HTML=$(find "$TARGET" -name "*_viewer.html" 2>/dev/null | head -1)
    if [ -n "$HTML" ]; then
        RELATIVE=$(echo "$HTML" | sed "s|${OUTPUT_BASE}/||")
        echo ""
        echo -e "  ${GREEN}✓ Viewer ready${RESET}"
        echo ""
        echo -e "  ${WHITE}View locally:${RESET}"
        echo -e "  ${CYAN}  http://localhost:8080/${RELATIVE}${RESET}"
        echo ""
        echo -e "  ${WHITE}View online:${RESET}"
        echo -e "  ${CYAN}  https://viewer.delfictus.com/${RELATIVE}${RESET}"
        echo ""
    fi
else
    echo -e "  ${RED}Viewer generation failed${RESET}"; exit 1
fi
