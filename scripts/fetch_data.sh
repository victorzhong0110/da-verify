#!/usr/bin/env bash
# Fetch the InfiAgent-DABench (DAEval) public validation set into data/daeval/.
# The data is CC BY-NC 4.0 (see NOTICE) and is intentionally NOT vendored in
# this repo — run this once after cloning.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEST="$ROOT/data/daeval"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

echo "Cloning InfiAgent (shallow)…"
git clone --depth 1 https://github.com/InfiAgent/InfiAgent.git "$TMP/infiagent" >/dev/null 2>&1
SRC="$TMP/infiagent/examples/DA-Agent"

mkdir -p "$DEST"
cp "$SRC/data/da-dev-questions.jsonl" "$DEST/"
cp "$SRC/data/da-dev-labels.jsonl"    "$DEST/"
cp -R "$SRC/data/da-dev-tables"       "$DEST/"
cp "$SRC/eval_closed_form.py"         "$DEST/OFFICIAL_eval_closed_form.py"

echo "Done -> $DEST"
echo "  questions: $(wc -l < "$DEST/da-dev-questions.jsonl") | labels: $(wc -l < "$DEST/da-dev-labels.jsonl") | tables: $(ls "$DEST/da-dev-tables" | wc -l)"
