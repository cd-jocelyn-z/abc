#!/bin/bash

echo ""
echo "[1/3] Create Directory"
mkdir -p data
echo "[2/3] Download Datasets"

BASE_URL="https://gitlab.huma-num.fr/kytym/korpusou/-/raw/main/datasets"

curl -L "${BASE_URL}/train.jsonl" -o data/train.jsonl
curl -L "${BASE_URL}/dev.jsonl" -o data/dev.jsonl
curl -L "${BASE_URL}/test.jsonl" -o data/test.jsonl

echo ""
echo "Download complete!"
ls -lh data/