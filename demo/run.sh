#!/bin/bash
# PRISM-4D Demo Run Menu
# Website palette: cyan #00e5ff, text #e8ecf0, secondary #8892a0, dim #556070

CYAN='\033[38;2;0;229;255m'
WHITE='\033[1;38;2;232;236;240m'
GRAY='\033[38;2;136;146;160m'
DIM='\033[38;2;85;96;112m'
GREEN='\033[32;1m'
YELLOW='\033[33;1m'
RED='\033[31;1m'
RESET='\033[0m'
DIVCYAN='\033[38;2;0;140;180m'

TOPO_DIR="/opt/prism4d/topologies"
OUTPUT_BASE="/opt/prism4d/output"
TMUX_CONF="/opt/prism4d/bin/tmux.conf"
MAX_OUTPUT_MB=2000

TOPO_ORDER=("1btl" "1ade" "3k5v" "3l15_chainA")
declare -A TOPOS
TOPOS[1btl]="1BTL — T4 Lysozyme (4073 atoms)"
TOPOS[1ade]="1ADE — Adenylate Kinase (13294 atoms)"
TOPOS[3k5v]="3K5V — β-Lactamase (9139 atoms)"
TOPOS[3l15_chainA]="3L15 — Chain A (3112 atoms)"

cleanup_output() {
    local current_mb=$(du -sm "$OUTPUT_BASE" 2>/dev/null | awk '{print $1}')
    [ -z "$current_mb" ] && return
    if [ "$current_mb" -gt "$MAX_OUTPUT_MB" ]; then
        echo -e "  ${YELLOW}⚠ Output storage: ${current_mb}MB / ${MAX_OUTPUT_MB}MB cap${RESET}"
        echo -e "  ${GRAY}Deleting oldest runs to free space...${RESET}"
        while [ "$current_mb" -gt "$((MAX_OUTPUT_MB / 2))" ]; do
            local oldest_dir=$(ls -dt "$OUTPUT_BASE"/*/ 2>/dev/null | tail -1)
            local oldest_log=$(ls -t "$OUTPUT_BASE"/*.log 2>/dev/null | tail -1)
            if [ -z "$oldest_dir" ] && [ -z "$oldest_log" ]; then break; fi
            if [ -n "$oldest_dir" ]; then
                local dirname=$(basename "$oldest_dir")
                local dir_mb=$(du -sm "$oldest_dir" 2>/dev/null | awk '{print $1}')
                rm -rf "$oldest_dir"
                rm -f "$OUTPUT_BASE/${dirname}.log" 2>/dev/null
                echo -e "  ${DIM}  Deleted: ${dirname} (${dir_mb}MB)${RESET}"
            elif [ -n "$oldest_log" ]; then
                rm -f "$oldest_log"
            fi
            current_mb=$(du -sm "$OUTPUT_BASE" 2>/dev/null | awk '{print $1}')
        done
        echo -e "  ${GREEN}  Storage now: ${current_mb}MB${RESET}"
    fi
}

browse_output() {
    local dirs=()
    local i=1
    while IFS= read -r d; do
        [ -d "$d" ] && dirs+=("$d")
    done < <(ls -dt "$OUTPUT_BASE"/*/ 2>/dev/null)

    if [ ${#dirs[@]} -eq 0 ]; then
        echo ""
        echo -e "  ${DIM}No previous runs found in output/${RESET}"
        return
    fi

    echo ""
    echo -e "${DIVCYAN}  ──────────────────────────────────────────${RESET}"
    echo -e "${WHITE}  Previous Runs:${RESET}"
    echo ""

    for d in "${dirs[@]}"; do
        local dirname=$(basename "$d")
        local dir_mb=$(du -sm "$d" 2>/dev/null | awk '{print $1}')
        local file_count=$(ls "$d" 2>/dev/null | wc -l)
        echo -e "  ${GREEN}${i})${RESET}  ${dirname}  ${DIM}(${dir_mb}MB, ${file_count} files)${RESET}"
        i=$((i + 1))
        [ $i -gt 20 ] && break
    done

    echo ""
    echo -e "  ${DIM}b)${RESET}  Back to menu"
    echo ""
    echo -n "  Select run to inspect: "
    read -r choice

    case "$choice" in
        b|B|"") return ;;
        *)
            if [[ "$choice" =~ ^[0-9]+$ ]] && [ "$choice" -ge 1 ] && [ "$choice" -le ${#dirs[@]} ]; then
                inspect_run "${dirs[$((choice - 1))]}"
            else
                echo -e "  ${RED}Invalid selection${RESET}"
            fi
            ;;
    esac
}

inspect_run() {
    local dir="$1"
    local dirname=$(basename "$dir")

    echo ""
    echo -e "${DIVCYAN}  ──────────────────────────────────────────${RESET}"
    echo -e "${WHITE}  Run: ${dirname}${RESET}"
    echo ""
    echo -e "  ${WHITE}Files:${RESET}"
    ls -lhS "$dir" | tail -n +2 | while read -r line; do
        echo -e "  ${GRAY}${line}${RESET}"
    done

    local md_file=$(find "$dir" -name "*.binding_sites.md" 2>/dev/null | head -1)
    local json_file=$(find "$dir" -name "*.binding_sites.json" 2>/dev/null | head -1)
    local pml_file=$(find "$dir" -name "*.pml" 2>/dev/null | head -1)
    local cxc_file=$(find "$dir" -name "*.cxc" 2>/dev/null | head -1)
    local log_file="${OUTPUT_BASE}/${dirname}.log"

    echo ""
    echo -e "${DIVCYAN}  ──────────────────────────────────────────${RESET}"
    echo -e "${WHITE}  View:${RESET}"
    echo ""

    local opts=()
    local opt_num=1

    if [ -n "$md_file" ]; then
        echo -e "  ${GREEN}${opt_num})${RESET}  Binding sites report (.md)"
        opts+=("md:$md_file"); opt_num=$((opt_num + 1))
    fi
    if [ -n "$json_file" ]; then
        echo -e "  ${GREEN}${opt_num})${RESET}  Binding sites data (.json)"
        opts+=("json:$json_file"); opt_num=$((opt_num + 1))
    fi
    if [ -f "$log_file" ]; then
        echo -e "  ${GREEN}${opt_num})${RESET}  Run log"
        opts+=("log:$log_file"); opt_num=$((opt_num + 1))
    fi
    [ -n "$pml_file" ] && echo -e "  ${CYAN}p)${RESET}  PyMOL script path"
    [ -n "$cxc_file" ] && echo -e "  ${CYAN}x)${RESET}  ChimeraX script path"

    echo ""
    echo -e "  ${DIM}b)${RESET}  Back"
    echo ""
    echo -n "  Select: "
    read -r view_choice

    case "$view_choice" in
        b|B|"") return ;;
        p|P) [ -n "$pml_file" ] && echo -e "\n  ${CYAN}PyMOL script:${RESET} ${pml_file}" ;;
        x|X) [ -n "$cxc_file" ] && echo -e "\n  ${CYAN}ChimeraX script:${RESET} ${cxc_file}" ;;
        *)
            if [[ "$view_choice" =~ ^[0-9]+$ ]] && [ "$view_choice" -ge 1 ] && [ "$view_choice" -lt "$opt_num" ]; then
                local selected="${opts[$((view_choice - 1))]}"
                local ftype="${selected%%:*}"
                local fpath="${selected#*:}"
                echo ""
                echo -e "${DIVCYAN}  ──────────────────────────────────────────${RESET}"
                case "$ftype" in
                    md)
                        echo -e "${WHITE}  Binding Sites Report${RESET}"
                        echo ""; cat "$fpath" ;;
                    json)
                        echo -e "${WHITE}  Binding Sites Data (summary)${RESET}"
                        echo ""; head -80 "$fpath"
                        local total_lines=$(wc -l < "$fpath")
                        [ "$total_lines" -gt 80 ] && echo -e "\n  ${DIM}... (${total_lines} total lines — use 'less ${fpath}' for full view)${RESET}" ;;
                    log)
                        echo -e "${WHITE}  Run Log (last 50 lines)${RESET}"
                        echo ""; tail -50 "$fpath"
                        echo -e "\n  ${DIM}Full log: less ${fpath}${RESET}" ;;
                esac
                echo ""
                echo -e "${DIVCYAN}  ──────────────────────────────────────────${RESET}"
            else
                echo -e "  ${RED}Invalid selection${RESET}"
            fi
            ;;
    esac
}

check_existing() {
    if tmux -f "$TMUX_CONF" has-session -t nhs 2>/dev/null; then
        echo ""
        echo -e "  ${YELLOW}⚡ An NHS run is still active!${RESET}"
        echo ""
        echo -e "  ${GREEN}1)${RESET}  Reattach to running session"
        echo -e "  ${RED}2)${RESET}  Kill it and start fresh"
        echo -e "  ${DIM}3)${RESET}  Back to menu"
        echo ""
        echo -n "  Select [1-3]: "
        read -r choice
        case "$choice" in
            1) tmux -f "$TMUX_CONF" attach -t nhs; return 0 ;;
            2)
                tmux -f "$TMUX_CONF" send-keys -t nhs C-c 2>/dev/null
                sleep 1
                tmux -f "$TMUX_CONF" kill-session -t nhs 2>/dev/null
                echo -e "  ${RED}Killed previous run${RESET}"
                return 1 ;;
            *) return 0 ;;
        esac
    fi
    return 1
}

print_header() {
    echo ""
    echo -e "${DIVCYAN}  ┌──────────────────────────────────────────┐${RESET}"
    echo -e "${DIVCYAN}  │${WHITE}     PRISM-4D  NHS Demo Runner            ${DIVCYAN}│${RESET}"
    echo -e "${DIVCYAN}  │${DIM}     Cryptic Binding Site Detection        ${DIVCYAN}│${RESET}"
    echo -e "${DIVCYAN}  └──────────────────────────────────────────┘${RESET}"
    echo ""
    local current_mb=$(du -sm "$OUTPUT_BASE" 2>/dev/null | awk '{print $1}')
    if [ -n "$current_mb" ] && [ "$current_mb" -gt 0 ]; then
        local run_count=$(ls -d "$OUTPUT_BASE"/*/ 2>/dev/null | wc -l)
        echo -e "  ${DIM}Storage: ${current_mb}MB / ${MAX_OUTPUT_MB}MB  (${run_count} runs saved)${RESET}"
        echo ""
    fi
}

print_presets() {
    echo -e "${WHITE}  Quick Presets:${RESET}"
    echo ""
    echo -e "  ${GREEN}1)${RESET}  1BTL — Quick   (4 streams, cutoff 8.0, fast)"
    echo -e "  ${GREEN}2)${RESET}  1BTL — Deep    (8 streams, cutoff 10.0, fast)"
    echo -e "  ${GREEN}3)${RESET}  3K5V — Quick   (4 streams, cutoff 8.0, fast)"
    echo -e "  ${GREEN}4)${RESET}  3K5V — Deep    (8 streams, cutoff 10.0, fast)"
    echo -e "  ${GREEN}5)${RESET}  1ADE — Quick   (4 streams, cutoff 8.0, fast)"
    echo -e "  ${GREEN}6)${RESET}  1ADE — Deep    (8 streams, cutoff 10.0, fast)"
    echo -e "  ${GREEN}7)${RESET}  3L15 — Quick   (4 streams, cutoff 8.0, fast)"
    echo -e "  ${GREEN}8)${RESET}  3L15 — Deep    (8 streams, cutoff 10.0, fast)"
    echo ""
    echo -e "${WHITE}  Custom:${RESET}"
    echo -e "  ${YELLOW}c)${RESET}  Custom configuration"
    echo ""
    echo -e "${WHITE}  Output:${RESET}"
    echo -e "  ${CYAN}o)${RESET}  Browse previous run results"
    echo ""
    echo -e "${WHITE}  Tools:${RESET}"
    echo -e "  ${CYAN}g)${RESET}  GPU stats (nvidia-smi)"
    echo -e "  ${CYAN}m)${RESET}  GPU monitor (nvitop)"
    echo ""
    echo -e "  ${DIM}q)${RESET}  Quit"
    echo ""
}

select_topology() {
    echo ""
    echo -e "${WHITE}  Select protein:${RESET}"
    for i in "${!TOPO_ORDER[@]}"; do
        local key="${TOPO_ORDER[$i]}"
        echo -e "  ${GREEN}$((i+1)))${RESET}  ${TOPOS[$key]}"
    done
    echo -n "  Select [1-4]: "
    read -r choice
    case "$choice" in
        1) SELECTED_TOPO="1btl" ;; 2) SELECTED_TOPO="1ade" ;;
        3) SELECTED_TOPO="3k5v" ;; 4) SELECTED_TOPO="3l15_chainA" ;;
        *) echo -e "  ${RED}Invalid${RESET}"; return 1 ;;
    esac
}

select_streams() {
    echo -e "${WHITE}  CUDA streams:${RESET}"
    echo -e "  ${GREEN}1)${RESET}  4 streams"
    echo -e "  ${GREEN}2)${RESET}  8 streams"
    echo -n "  Select [1-2]: "
    read -r choice
    case "$choice" in 1) SELECTED_STREAMS=4 ;; 2) SELECTED_STREAMS=8 ;; *) SELECTED_STREAMS=4 ;; esac
}

select_cutoff() {
    echo -e "${WHITE}  Lining cutoff (Å):${RESET}"
    echo -e "  ${GREEN}1)${RESET}  8.0  (standard)"
    echo -e "  ${GREEN}2)${RESET}  9.0  (extended)"
    echo -e "  ${GREEN}3)${RESET}  10.0 (deep)"
    echo -n "  Select [1-3]: "
    read -r choice
    case "$choice" in 1) SELECTED_CUTOFF="8.0" ;; 2) SELECTED_CUTOFF="9.0" ;; 3) SELECTED_CUTOFF="10.0" ;; *) SELECTED_CUTOFF="8.0" ;; esac
}

select_steps() {
    echo -e "${WHITE}  Simulation steps:${RESET}"
    echo -e "  ${GREEN}1)${RESET}  35,000   (fast ~30s)"
    echo -e "  ${GREEN}2)${RESET}  100,000  (medium ~1-2 min)"
    echo -e "  ${GREEN}3)${RESET}  250,000  (thorough ~3-5 min)"
    echo -e "  ${GREEN}4)${RESET}  500,000  (full ~5-10 min)"
    echo -n "  Select [1-4]: "
    read -r choice
    case "$choice" in 1) SELECTED_STEPS=35000 ;; 2) SELECTED_STEPS=100000 ;; 3) SELECTED_STEPS=250000 ;; 4) SELECTED_STEPS=500000 ;; *) SELECTED_STEPS=500000 ;; esac
}

select_fast() {
    echo -e "${WHITE}  Protocol:${RESET}"
    echo -e "  ${GREEN}1)${RESET}  Fast (35K high-energy UV)"
    echo -e "  ${GREEN}2)${RESET}  Standard (full thermal)"
    echo -n "  Select [1-2]: "
    read -r choice
    case "$choice" in 1) SELECTED_FAST="--fast" ;; 2) SELECTED_FAST="" ;; *) SELECTED_FAST="--fast" ;; esac
}

run_nhs() {
    local topo="$1" streams="$2" cutoff="$3" steps="$4" fast="$5"
    local topo_file="${TOPO_DIR}/${topo}.topology.json"
    local timestamp=$(date +%Y%m%d_%H%M%S)
    local out_dir="${OUTPUT_BASE}/${topo}_${timestamp}"
    local log_file="${OUTPUT_BASE}/${topo}_${timestamp}.log"

    if [ ! -f "$topo_file" ]; then
        echo -e "  ${RED}ERROR: Topology not found: ${topo_file}${RESET}"
        return 1
    fi

    cleanup_output
    mkdir -p "$out_dir"

    echo ""
    echo -e "${DIVCYAN}  ──────────────────────────────────────────${RESET}"
    echo -e "${WHITE}  Configuration:${RESET}"
    echo -e "  ${DIM}Protein:${RESET}    ${topo}"
    echo -e "  ${DIM}Streams:${RESET}    ${streams}"
    echo -e "  ${DIM}Cutoff:${RESET}     ${cutoff} Å"
    echo -e "  ${DIM}Steps:${RESET}      ${steps}"
    echo -e "  ${DIM}Protocol:${RESET}   $([ -n "$fast" ] && echo "Fast UV" || echo "Standard")"
    echo -e "  ${DIM}Output:${RESET}     ${out_dir}"
    echo -e "  ${DIM}Log:${RESET}        ${log_file}"
    echo -e "${DIVCYAN}  ──────────────────────────────────────────${RESET}"
    echo ""

    local nhs_cmd="RUST_LOG=info nhs_rt_full -t ${topo_file} -o ${out_dir} --multi-stream ${streams} --multi-scale --rt-clustering --lining-cutoff ${cutoff} --steps ${steps} ${fast} -v"

    echo -e "  ${DIM}\$ ${nhs_cmd}${RESET}"
    echo ""
    echo -e "  ${WHITE}Stop run:${RESET}  type ${RED}q${WHITE} + Enter"
    echo -e "  ${WHITE}Detach:${RESET}    type ${CYAN}d${WHITE} + Enter  (run keeps going)"
    echo -e "  ${WHITE}Reattach:${RESET}  type ${CYAN}run${WHITE} from main shell"
    echo ""
    sleep 2

    tmux -f "$TMUX_CONF" kill-session -t nhs 2>/dev/null

    cat > /tmp/nhs_wrapper.sh << WRAPPER
#!/bin/bash
export PRISM4D_PTX_DIR=/opt/prism4d/kernels/ptx
export PRISM_PTX_DIR=/opt/prism4d/kernels/ptx
export PRISM_OPTIXIR_DIR=/opt/prism4d/kernels/optixir
export PATH=/opt/prism4d/bin:\$PATH

CYAN='\033[38;2;0;229;255m'
WHITE='\033[1;38;2;232;236;240m'
GRAY='\033[38;2;136;146;160m'
DIM='\033[38;2;85;96;112m'
GREEN='\033[32;1m'
YELLOW='\033[33;1m'
RED='\033[31;1m'
RESET='\033[0m'
DIVCYAN='\033[38;2;0;140;180m'

OUT_DIR="${out_dir}"
LOG_FILE="${log_file}"

${nhs_cmd} > "\$LOG_FILE" 2>&1 &
NHS_PID=\$!

tail -f "\$LOG_FILE" &
TAIL_PID=\$!

echo ""
echo -e "\${YELLOW}━━━ NHS running (PID \$NHS_PID) ━━━\${RESET}"
echo -e "\${WHITE}  q + Enter = stop    d + Enter = detach\${RESET}"
echo ""

while kill -0 "\$NHS_PID" 2>/dev/null; do
    if read -t 1 -r input; then
        case "\$input" in
            q|Q|quit|QUIT|stop|STOP)
                echo ""
                echo -e "  \${RED}⚠ Stopping NHS (PID \$NHS_PID)...\${RESET}"
                kill "\$NHS_PID" 2>/dev/null; sleep 1
                kill -9 "\$NHS_PID" 2>/dev/null
                wait "\$NHS_PID" 2>/dev/null
                kill "\$TAIL_PID" 2>/dev/null
                echo -e "  \${RED}✗ Run cancelled by user\${RESET}"
                echo -e "\${DIVCYAN}  ──────────────────────────────────────────\${RESET}"
                echo -e "  \${DIM}Log saved: \${LOG_FILE}\${RESET}"
                echo -e "  \${DIM}Press Enter to close...\${RESET}"
                read -r; exit 0 ;;
            d|D|detach|DETACH)
                echo ""
                echo -e "  \${CYAN}Detaching — NHS keeps running in background\${RESET}"
                echo -e "  \${CYAN}Type 'run' to reattach\${RESET}"
                sleep 1; tmux detach-client ;;
        esac
    fi
done

wait "\$NHS_PID"; EXIT_CODE=\$?
kill "\$TAIL_PID" 2>/dev/null

echo ""
echo -e "\${DIVCYAN}  ──────────────────────────────────────────\${RESET}"
if [ \$EXIT_CODE -eq 0 ]; then
    echo -e "  \${GREEN}✓ Run complete\${RESET}"
else
    echo -e "  \${RED}✗ Exited with code \${EXIT_CODE}\${RESET}"
fi

if [ -d "\$OUT_DIR" ] && [ "\$(ls -A \$OUT_DIR 2>/dev/null)" ]; then
    echo ""
    echo -e "  \${WHITE}Output files:\${RESET}"
    ls -lh "\$OUT_DIR" | tail -n +2 | while read -r line; do
        echo -e "  \${GRAY}\${line}\${RESET}"
    done
    HTML=\$(find "\$OUT_DIR" -name "*.html" 2>/dev/null)
    if [ -n "\$HTML" ]; then
        echo ""; echo -e "  \${GREEN}Interactive visualization:\${RESET}"
        echo "\$HTML" | while read -r f; do echo -e "  \${CYAN}\${f}\${RESET}"; done
    fi
    JSON=\$(find "\$OUT_DIR" -name "*.json" -o -name "*report*" -o -name "*summary*" 2>/dev/null)
    if [ -n "\$JSON" ]; then
        echo ""; echo -e "  \${GREEN}Reports:\${RESET}"
        echo "\$JSON" | while read -r f; do echo -e "  \${CYAN}\${f}\${RESET}"; done
    fi
else
    echo -e "  \${DIM}No output files generated\${RESET}"
fi

echo -e "\${DIVCYAN}  ──────────────────────────────────────────\${RESET}"
echo ""; echo -e "  \${DIM}Log: \${LOG_FILE}\${RESET}"
echo -e "  \${DIM}Press Enter to close...\${RESET}"; read -r
WRAPPER
    chmod +x /tmp/nhs_wrapper.sh

    tmux -f "$TMUX_CONF" new-session -d -s nhs -x 200 -y 50 "bash /tmp/nhs_wrapper.sh"
    tmux -f "$TMUX_CONF" attach -t nhs
}

custom_config() {
    select_topology || return
    select_streams; select_cutoff; select_steps; select_fast
    run_nhs "$SELECTED_TOPO" "$SELECTED_STREAMS" "$SELECTED_CUTOFF" "$SELECTED_STEPS" "$SELECTED_FAST"
}

# Handle CLI arguments
if [ "$1" = "output" ] || [ "$1" = "o" ]; then
    browse_output; exit 0
fi

while true; do
    if tmux -f "$TMUX_CONF" has-session -t nhs 2>/dev/null; then
        check_existing; ret=$?; [ $ret -eq 0 ] && continue
    fi

    print_header; print_presets
    echo -n "  Select: "
    read -r selection
    case "$selection" in
        1) run_nhs "1btl" 4 "8.0" 500000 "--fast" ;;
        2) run_nhs "1btl" 8 "10.0" 500000 "--fast" ;;
        3) run_nhs "3k5v" 4 "8.0" 500000 "--fast" ;;
        4) run_nhs "3k5v" 8 "10.0" 500000 "--fast" ;;
        5) run_nhs "1ade" 4 "8.0" 500000 "--fast" ;;
        6) run_nhs "1ade" 8 "10.0" 500000 "--fast" ;;
        7) run_nhs "3l15_chainA" 4 "8.0" 500000 "--fast" ;;
        8) run_nhs "3l15_chainA" 8 "10.0" 500000 "--fast" ;;
        c|C) custom_config ;;
        o|O|output|OUTPUT) browse_output ;;
        g|G) echo ""; nvidia-smi; echo "" ;;
        m|M) nvitop ;;
        q|Q) echo -e "  ${DIM}Goodbye.${RESET}"; exit 0 ;;
        *) echo -e "  ${RED}Invalid selection${RESET}" ;;
    esac
    echo ""
    echo -e "  ${DIM}Press Enter to return to menu...${RESET}"
    read -r
done
