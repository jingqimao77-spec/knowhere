#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname "$0")" >/dev/null 2>&1 && pwd -P)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
OUT_DIR="$ROOT_DIR/output/unittest"
SIFT_FILE="$OUT_DIR/sift-128-euclidean.hdf5"
REF_MAKEFILE="$ROOT_DIR/benchmark/hdf5/ref_logs/Makefile"

mkdir -p "$OUT_DIR"
if [[ ! -f "$SIFT_FILE" ]]; then
    wget -O "$SIFT_FILE" https://ann-benchmarks.com/sift-128-euclidean.hdf5
fi
if [[ -f "$REF_MAKEFILE" ]]; then
    cp "$REF_MAKEFILE" "$OUT_DIR/"
fi
