#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname "$0")" >/dev/null 2>&1 && pwd -P)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
BUILD_DIR="${BUILD_DIR:-$ROOT_DIR/build/Benchmark}"
DATASET_PATH="${1:-$ROOT_DIR/output/unittest/sift-128-euclidean.hdf5}"
WORK_DIR="${2:-$BUILD_DIR/benchmark/hdf5/diskann_test}"
BIN="$BUILD_DIR/benchmark/diskann_mmap_eval"
INDEX_DIR="$WORK_DIR/l2_index"
DOC_PATH="${DOC_PATH:-$ROOT_DIR/docs/diskann_mmap_evaluation.md}"
APPEND_DOC="${APPEND_DOC:-1}"
LD_PATHS="$BUILD_DIR:$BUILD_DIR/generators:$ROOT_DIR/build/Release:$ROOT_DIR/build/Release/generators"
TMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TMP_DIR"' EXIT

metric_from_log() {
    local file="$1"
    local key="$2"
    awk -F= -v key="$key" '$1 == key {print $2}' "$file" | tail -n 1
}

residency_from_log() {
    local file="$1"
    awk '/Resident Pages:/{sub(/^  */, ""); print; exit}' "$file"
}

append_results_to_doc() {
    if [[ "$APPEND_DOC" != "1" ]]; then
        return 0
    fi
    if [[ ! -f "$DOC_PATH" ]]; then
        echo "document not found, skip append: $DOC_PATH" >&2
        return 0
    fi

    local ts
    ts="$(date -u '+%Y-%m-%d %H:%M:%S UTC')"

    local deserialize_0 search_0 recall_0 rss_load_0 rss_search_0 hwm_0
    local deserialize_1 search_1 recall_1 rss_load_1 rss_search_1 hwm_1
    local residency_0 residency_1

    deserialize_0="$(metric_from_log "$TMP_DIR/bench_0.log" "deserialize_ms")"
    search_0="$(metric_from_log "$TMP_DIR/bench_0.log" "search_ms")"
    recall_0="$(metric_from_log "$TMP_DIR/bench_0.log" "recall_at_100")"
    rss_load_0="$(metric_from_log "$TMP_DIR/bench_0.log" "rss_after_load_kb")"
    rss_search_0="$(metric_from_log "$TMP_DIR/bench_0.log" "rss_after_search_kb")"
    hwm_0="$(metric_from_log "$TMP_DIR/bench_0.log" "vmhwm_kb")"

    deserialize_1="$(metric_from_log "$TMP_DIR/bench_1.log" "deserialize_ms")"
    search_1="$(metric_from_log "$TMP_DIR/bench_1.log" "search_ms")"
    recall_1="$(metric_from_log "$TMP_DIR/bench_1.log" "recall_at_100")"
    rss_load_1="$(metric_from_log "$TMP_DIR/bench_1.log" "rss_after_load_kb")"
    rss_search_1="$(metric_from_log "$TMP_DIR/bench_1.log" "rss_after_search_kb")"
    hwm_1="$(metric_from_log "$TMP_DIR/bench_1.log" "vmhwm_kb")"

    residency_0="$(residency_from_log "$TMP_DIR/evict_0.log")"
    residency_1="$(residency_from_log "$TMP_DIR/evict_1.log")"

    cat >> "$DOC_PATH" <<DOC

## Formal Benchmark Run: $ts

### Scope

- entry script: benchmark/run_diskann_mmap_eval.sh
- benchmark target: benchmark/diskann_mmap_eval
- dataset: $DATASET_PATH
- work directory: $WORK_DIR
- index directory: $INDEX_DIR

### Cache Control

- use_mmap=0 residency after eviction: $residency_0
- use_mmap=1 residency after eviction: $residency_1

### Measured Result

| Mode | deserialize_ms | search_ms | recall_at_100 | rss_after_load_kb | rss_after_search_kb | vmhwm_kb |
|---|---:|---:|---:|---:|---:|---:|
| use_mmap=0 | $deserialize_0 | $search_0 | $recall_0 | $rss_load_0 | $rss_search_0 | $hwm_0 |
| use_mmap=1 | $deserialize_1 | $search_1 | $recall_1 | $rss_load_1 | $rss_search_1 | $hwm_1 |
DOC

    echo "appended benchmark result to $DOC_PATH" >&2
}

if ! command -v vmtouch >/dev/null 2>&1; then
    echo "vmtouch is required but not installed" >&2
    exit 1
fi

if [[ ! -f "$DATASET_PATH" ]]; then
    echo "dataset not found: $DATASET_PATH" >&2
    echo "downloading with benchmark/prepare.sh" >&2
    "$SCRIPT_DIR/prepare.sh"
fi

if [[ ! -x "$BIN" ]]; then
    echo "building diskann_mmap_eval target in $BUILD_DIR" >&2
    cmake --build "$BUILD_DIR" --target diskann_mmap_eval -j"$(nproc)"
fi

mkdir -p "$WORK_DIR"

if [[ ! -f "$INDEX_DIR/l2_disk.index" ]]; then
    echo "preparing DiskANN index under $WORK_DIR" >&2
    LD_LIBRARY_PATH="$LD_PATHS:${LD_LIBRARY_PATH:-}" "$BIN" prepare "$DATASET_PATH" "$WORK_DIR"
fi

run_mode() {
    local mode="$1"
    local evict_log="$TMP_DIR/evict_${mode}.log"
    local bench_log="$TMP_DIR/bench_${mode}.log"

    {
        echo "=== evicting DiskANN index files for use_mmap=$mode ==="
        vmtouch -e "$INDEX_DIR"/*
        echo "--- residency after eviction ---"
        vmtouch "$INDEX_DIR"/*
    } | tee "$evict_log"

    echo "--- benchmark use_mmap=$mode ---"
    LD_LIBRARY_PATH="$LD_PATHS:${LD_LIBRARY_PATH:-}" "$BIN" bench "$DATASET_PATH" "$WORK_DIR" "$mode" | tee "$bench_log"
}

run_mode 0
run_mode 1
append_results_to_doc
