#!/usr/bin/env bash
# archive_dead_artifacts.sh
# Archive dead/orphan training artifacts to free disk space.
#
# Targets (verified 2026-05-04 audit):
#   - output/micro-kiki/stacks-v3-r16/        (32 stacks, lora_B=0, val identical = collapsed)
#   - output/micro-kiki/lora-qwen36-35b-hybrid/ (empty subdirs)
#   - output/micro-kiki/stack-01-chat-fr-v2/  (config only, no weights)
#   - output/qwen35-122b-macport/             (config only, training crashed OOM)
#   - output/qwen35-35b-opus-14k-v1/          (config only, never trained)
#   - output/qwen35-35b-opus-final/           (config only, never trained)
#
# Default: DRY-RUN. Pass --execute to actually move.
# Move target: $REPO_ROOT/_archive/<YYYY-MM-DD>/<original-relative-path>
#
# Safety:
#   - Refuse to run if any target is missing (use --skip-missing to override)
#   - Prints sizes before/after
#   - Refuse to run if a process is currently writing to the target
#   - Logs every operation to _archive/<date>/archive.log

set -euo pipefail

# --- Resolve repo root (script is in tools/) ---
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

# --- CLI ---
EXECUTE=0
SKIP_MISSING=0
for arg in "$@"; do
    case "$arg" in
        --execute) EXECUTE=1 ;;
        --skip-missing) SKIP_MISSING=1 ;;
        --help|-h)
            head -n 25 "$0" | sed 's/^# //'
            exit 0
            ;;
        *)
            echo "Unknown option: $arg" >&2
            echo "Usage: $0 [--execute] [--skip-missing]" >&2
            exit 2
            ;;
    esac
done

DATE="$(date +%Y-%m-%d)"
ARCHIVE_ROOT="_archive/$DATE"
LOGFILE="$ARCHIVE_ROOT/archive.log"

# --- Targets (relative to repo root) ---
TARGETS=(
    "output/micro-kiki/stacks-v3-r16"
    "output/micro-kiki/lora-qwen36-35b-hybrid"
    "output/micro-kiki/stack-01-chat-fr-v2"
    "output/qwen35-122b-macport"
    "output/qwen35-35b-opus-14k-v1"
    "output/qwen35-35b-opus-final"
)

# --- Pretty printing ---
if [ -t 1 ]; then
    BOLD="$(tput bold)"; DIM="$(tput dim)"; RED="$(tput setaf 1)"; GREEN="$(tput setaf 2)"; YELLOW="$(tput setaf 3)"; CYAN="$(tput setaf 6)"; RESET="$(tput sgr0)"
else
    BOLD=""; DIM=""; RED=""; GREEN=""; YELLOW=""; CYAN=""; RESET=""
fi

log() {
    local msg="$1"
    echo "$msg"
    if [ "$EXECUTE" = "1" ]; then
        printf '[%s] %s\n' "$(date +%H:%M:%S)" "$msg" | sed 's/\x1b\[[0-9;]*m//g' >> "$LOGFILE"
    fi
}

human_size() {
    local path="$1"
    if [ -e "$path" ]; then
        du -sh "$path" 2>/dev/null | awk '{print $1}'
    else
        echo "—"
    fi
}

# --- Header ---
echo
echo "${BOLD}archive_dead_artifacts.sh${RESET}  ${DIM}(repo: $REPO_ROOT)${RESET}"
if [ "$EXECUTE" = "1" ]; then
    echo "${BOLD}${RED}MODE: EXECUTE${RESET} — files will be moved"
else
    echo "${BOLD}${YELLOW}MODE: DRY-RUN${RESET} — pass ${BOLD}--execute${RESET} to actually move"
fi
echo "${DIM}Archive root: $ARCHIVE_ROOT${RESET}"
echo

# --- Validate targets ---
MISSING=()
PRESENT=()
for t in "${TARGETS[@]}"; do
    if [ -e "$t" ]; then
        PRESENT+=("$t")
    else
        MISSING+=("$t")
    fi
done

if [ ${#MISSING[@]} -gt 0 ]; then
    echo "${YELLOW}Missing targets:${RESET}"
    for t in "${MISSING[@]}"; do echo "  - $t"; done
    echo
    if [ "$SKIP_MISSING" = "0" ] && [ ${#PRESENT[@]} -eq 0 ]; then
        echo "${RED}No targets present and --skip-missing not set. Aborting.${RESET}"
        exit 1
    fi
    if [ "$SKIP_MISSING" = "0" ]; then
        echo "${RED}Some targets are missing. Pass --skip-missing to proceed with present ones only.${RESET}"
        exit 1
    fi
fi

if [ ${#PRESENT[@]} -eq 0 ]; then
    echo "${GREEN}Nothing to archive — all targets already absent.${RESET}"
    exit 0
fi

# --- Inventory ---
echo "${BOLD}Targets to archive:${RESET}"
TOTAL_BYTES=0
for t in "${PRESENT[@]}"; do
    size_h="$(human_size "$t")"
    size_b="$(du -sk "$t" 2>/dev/null | awk '{print $1*1024}')"
    TOTAL_BYTES=$((TOTAL_BYTES + size_b))
    nfiles="$(find "$t" -type f 2>/dev/null | wc -l | tr -d ' ')"
    echo "  ${CYAN}$t${RESET}  ${DIM}(${size_h}, ${nfiles} files)${RESET}"
done
echo
TOTAL_HUMAN="$(awk -v b=$TOTAL_BYTES 'BEGIN{
    split("B KB MB GB TB", u);
    i=1; while (b>=1024 && i<5) { b/=1024; i++ }
    printf("%.1f %s", b, u[i])
}')"
echo "${BOLD}Total to archive: $TOTAL_HUMAN${RESET}"
echo

# --- Safety: check for write locks (any process with open fds) ---
for t in "${PRESENT[@]}"; do
    if command -v lsof >/dev/null 2>&1; then
        if lsof +D "$t" 2>/dev/null | grep -q .; then
            echo "${RED}ERROR: '$t' is currently in use by another process. Aborting.${RESET}"
            lsof +D "$t" | head -5
            exit 3
        fi
    fi
done

# --- Execute or dry-run ---
if [ "$EXECUTE" = "0" ]; then
    echo "${YELLOW}Dry-run complete. No files moved.${RESET}"
    echo "${DIM}Re-run with --execute to actually archive.${RESET}"
    exit 0
fi

mkdir -p "$ARCHIVE_ROOT"
: > "$LOGFILE"
log "Archive operation started by $(whoami) on $(hostname)"
log "Repo root: $REPO_ROOT"
log "Total to archive: $TOTAL_HUMAN"

for t in "${PRESENT[@]}"; do
    dest="$ARCHIVE_ROOT/$t"
    dest_parent="$(dirname "$dest")"
    mkdir -p "$dest_parent"
    log "${GREEN}MV${RESET} $t  →  $dest"
    mv "$t" "$dest"
done

log "Done. Archived to $ARCHIVE_ROOT"
echo
echo "${BOLD}${GREEN}Archive complete.${RESET}"
echo "${DIM}To restore: mv $ARCHIVE_ROOT/<target> <target>${RESET}"
echo "${DIM}To free space permanently: rm -rf $ARCHIVE_ROOT/${RESET}"
echo "${DIM}Log: $LOGFILE${RESET}"
