#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MAIN_BIN="${MAIN_BIN:-$ROOT_DIR/bin/main}"

if [[ ! -x "$MAIN_BIN" ]]; then
  echo "ERROR: binary not found or not executable: $MAIN_BIN"
  echo "Run 'make' from the repository root first."
  exit 1
fi

# Argument order:
# seed N_sc N_pb N_tp N_it N_pn beta lost al alphaScale wht FixMode DynamicFix viMode vfMode evalMode

# Small AIR-B&B example (wht=30). ApproxGapEps is fixed internally in Main.cpp.
"$MAIN_BIN" \
  1 5 5 6 2 2 \
  1.0 30 0.5 0.5 \
  30 1 0 0 0 1

# Optional Ma-style enhanced branch-and-bound example (wht=24):
# "$MAIN_BIN" \
#   1 5 5 6 2 2 \
#   1.0 30 0.5 0.5 \
#   24 1 0 0 0 0
