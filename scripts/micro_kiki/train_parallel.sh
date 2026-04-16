#!/usr/bin/env bash
set -uo pipefail

# Train remaining Brainstacks stacks in parallel groups of 3
CONFIG="$(cd "$(dirname "$0")/../.." && pwd)/configs/micro_kiki/brainstacks.yaml"
SCRIPT="$(cd "$(dirname "$0")" && pwd)/train_stack.py"
LOG_DIR="$(cd "$(dirname "$0")/../.." && pwd)/output/micro-kiki/logs"
STACKS_DIR="$(cd "$(dirname "$0")/../.." && pwd)/output/micro-kiki/stacks"
PARALLEL=${1:-3}

mkdir -p "$LOG_DIR"

# Full curriculum with index
DOMAINS=(
  chat-fr reasoning python typescript cpp rust
  html-css shell sql yaml-json docker kicad-dsl
  spice lua-upy embedded stm32 iot freecad
  platformio power emc dsp spice-sim electronics
  kicad-pcb web-frontend web-backend music-audio
  devops llm-orch math security
)

# Find remaining stacks
REMAINING=()
REMAINING_IDX=()
for i in "${!DOMAINS[@]}"; do
  domain="${DOMAINS[$i]}"
  idx=$((i + 1))
  if [ ! -f "$STACKS_DIR/$domain/adapters.safetensors" ]; then
    REMAINING+=("$domain")
    REMAINING_IDX+=("$idx")
  fi
done

echo "================================================================"
echo "Brainstacks Parallel Training (${PARALLEL} at a time)"
echo "================================================================"
echo "Completed: $((32 - ${#REMAINING[@]}))/32"
echo "Remaining: ${#REMAINING[@]} stacks"
echo "Parallel:  ${PARALLEL}"
echo "================================================================"

TOTAL_START=$(date +%s)
DONE=0

# Process in groups
for ((g=0; g<${#REMAINING[@]}; g+=PARALLEL)); do
  PIDS=()
  GROUP_DOMAINS=()

  for ((j=0; j<PARALLEL && g+j<${#REMAINING[@]}; j++)); do
    idx_in_remaining=$((g + j))
    domain="${REMAINING[$idx_in_remaining]}"
    stack_idx="${REMAINING_IDX[$idx_in_remaining]}"
    GROUP_DOMAINS+=("$domain")

    echo ""
    echo "[$(( DONE + j + 1 ))/${#REMAINING[@]}] Launching: $domain (stack $stack_idx)"

    PYTHONUNBUFFERED=1 uv run python "$SCRIPT" \
      --config "$CONFIG" \
      --domain "$domain" \
      --stack-index "$stack_idx" \
      > "$LOG_DIR/${stack_idx}-${domain}.log" 2>&1 &
    PIDS+=($!)
  done

  echo ""
  echo "--- Waiting for group: ${GROUP_DOMAINS[*]} ---"

  # Wait for all in group
  FAILURES=0
  for ((j=0; j<${#PIDS[@]}; j++)); do
    wait "${PIDS[$j]}"
    EXIT=$?
    domain="${GROUP_DOMAINS[$j]}"
    DONE=$((DONE + 1))
    if [ $EXIT -eq 0 ]; then
      echo "  ✓ $domain complete ($DONE/${#REMAINING[@]})"
    else
      echo "  ✗ $domain FAILED (exit $EXIT) — see $LOG_DIR/*-${domain}.log"
      FAILURES=$((FAILURES + 1))
    fi
  done

  if [ $FAILURES -gt 0 ]; then
    echo "WARNING: $FAILURES failures in this group"
  fi
done

TOTAL_END=$(date +%s)
TOTAL_ELAPSED=$(( TOTAL_END - TOTAL_START ))
echo ""
echo "================================================================"
echo "All done in $((TOTAL_ELAPSED / 60))m$((TOTAL_ELAPSED % 60))s"
echo "Stacks completed:"
ls "$STACKS_DIR"/*/adapters.safetensors 2>/dev/null | wc -l
echo "================================================================"
