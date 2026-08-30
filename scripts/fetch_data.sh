#!/usr/bin/env bash
# Fetch the InfiAgent-DABench (DAEval) public validation set into data/daeval/.
# The data is CC BY-NC 4.0 (see NOTICE) and is intentionally NOT vendored in
# this repo — run this once after cloning.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEST="$ROOT/data/daeval"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

# Pin the upstream commit so register clones are reproducible.
INFIAGENT_COMMIT="3d6c4a70198e0a41fadf539f5b43c88b8c1a2d9c"

echo "Cloning InfiAgent @ ${INFIAGENT_COMMIT}…"
git init "$TMP/infiagent" >/dev/null
git -C "$TMP/infiagent" remote add origin https://github.com/InfiAgent/InfiAgent.git
git -C "$TMP/infiagent" fetch --depth 1 origin "$INFIAGENT_COMMIT"
git -C "$TMP/infiagent" checkout --detach FETCH_HEAD >/dev/null
SRC="$TMP/infiagent/examples/DA-Agent"

mkdir -p "$DEST"
cp "$SRC/data/da-dev-questions.jsonl" "$DEST/"
cp "$SRC/data/da-dev-labels.jsonl"    "$DEST/"
rm -rf "$DEST/da-dev-tables"
cp -R "$SRC/data/da-dev-tables"       "$DEST/"
cp "$SRC/eval_closed_form.py"         "$DEST/OFFICIAL_eval_closed_form.py"

echo "Done -> $DEST"
echo "  questions: $(wc -l < "$DEST/da-dev-questions.jsonl") | labels: $(wc -l < "$DEST/da-dev-labels.jsonl") | tables: $(ls "$DEST/da-dev-tables" | wc -l)"
echo "  infiagent: ${INFIAGENT_COMMIT}"
