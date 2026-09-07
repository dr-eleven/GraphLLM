#!/bin/bash

# Try to download the three GOFA evaluation .pt files that evaluation/evaluate_trained_model.py expects.
# The script first attempts an exact Hugging Face dataset download, then copies the files into place.

set -euo pipefail

ROOT_DIR="/opt/data/private/GraphSSR-master"
SNAPSHOT_DIR="$ROOT_DIR/.downloads/hf_dataset_snapshot"

FILES=(
  "data/test_data/raw_data/products/products_graph_bert_encoded.pt"
  "data/test_data/raw_data/wikics/wikics_graph_bert_encoded.pt"
  "data/test_data/raw_data/fb15k237/fb15k237_graph_bert_encoded.pt"
)

mkdir -p "$SNAPSHOT_DIR"

echo "Downloading from Hugging Face dataset: wfdlfz/GraphSSR"
huggingface-cli download wfdlfz/GraphSSR \
  --repo-type dataset \
  --include "data/test_data/raw_data/products/products_graph_bert_encoded.pt" \
  --include "data/test_data/raw_data/wikics/wikics_graph_bert_encoded.pt" \
  --include "data/test_data/raw_data/fb15k237/fb15k237_graph_bert_encoded.pt" \
  --local-dir "$SNAPSHOT_DIR"

for rel_path in "${FILES[@]}"; do
  src="$SNAPSHOT_DIR/$rel_path"
  dst="$ROOT_DIR/$rel_path"
  if [[ ! -f "$src" ]]; then
    echo "Missing after download: $src"
    continue
  fi
  mkdir -p "$(dirname "$dst")"
  cp "$src" "$dst"
  echo "Installed: $dst"
done

echo "Verification:"
for rel_path in "${FILES[@]}"; do
  if [[ -f "$ROOT_DIR/$rel_path" ]]; then
    echo "OK  $ROOT_DIR/$rel_path"
  else
    echo "MISS $ROOT_DIR/$rel_path"
  fi
done

