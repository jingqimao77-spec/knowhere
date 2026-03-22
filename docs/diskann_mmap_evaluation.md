# DiskANN mmap Evaluation

## Purpose

This document records how DiskANN mmap changes are evaluated in this repository.
It is intended to be appended after each implementation stage so we can compare:

- performance impact
- memory impact
- correctness and stability status

Current status:

- Stage 1 implemented
- Only `pq_compressed_vectors` is mmap-backed
- Small metadata files and `disk.index` are still loaded the original way

## Scope By Stage

### Stage 1

- `use_mmap` config added to DiskANN deserialize path
- `pq_compressed_vectors` can be loaded by mmap
- search path still uses the same logic
- `disk.index` still uses `AlignedFileReader`
- medoids / centroids / norms are not mmap-backed yet

### Planned Later Stages

- Stage 2: mmap small metadata files
- Stage 3: evaluate mmap for more search-time data paths
- Stage 4+: evaluate full mmap or hybrid provider-based design

## Evaluation Method

### 1. Correctness Validation

Use existing DiskANN unit tests with both:

- `use_mmap = false`
- `use_mmap = true`

Validation covers:

- `Deserialize`
- `Search`
- `RangeSearch`
- `GetVectorByIds`

Current verification command:

```bash
cmake --build build/Release --target knowhere_tests -j2
export LD_LIBRARY_PATH=/workspace/milvus/knowhere/build/Release:/workspace/milvus/knowhere/build/generators:${LD_LIBRARY_PATH:-}
/workspace/milvus/knowhere/build/Release/tests/ut/knowhere_tests "[diskann]"
```

### 2. Performance And Memory Benchmark

A temporary standalone benchmark program was used to isolate:

- `Deserialize` latency
- repeated `Search` latency
- memory after load
- memory after repeated search
- peak RSS

Benchmark setup used for Stage 1:

- dataset size: `50,000 x 512` fp32
- metric: `L2`
- `k = 10`
- search queries: `200`
- search rounds: `20`
- `search_cache_budget_gb = 0`
- compare only:
  - `use_mmap = 0`
  - `use_mmap = 1`

Reason for this setup:

- the PQ compressed file is large enough to show a real memory difference
- `Deserialize` cost is measurable
- search is repeated enough to observe steady-state behavior

## Evaluation Metrics

### Correctness

- unit tests pass / fail
- search recall unchanged
- no crash on repeated deserialize / search

### Performance

- `deserialize_ms`
  - wall-clock time for `Deserialize`
- `avg_search_ms`
  - average latency of repeated search rounds

### Memory

- `rss_after_load_kb`
  - RSS immediately after `Deserialize`
- `rss_after_search_kb`
  - RSS after repeated search rounds
- `vmhwm_kb`
  - process peak RSS (`VmHWM`)

## Stage 1 Result

### Build And Test Status

- `[diskann]` unit tests: passed
- result:

```text
All tests passed (12653266 assertions in 3 test cases)
```

### Benchmark Files

Observed benchmark index file sizes:

- `bench_pq_compressed.bin`: about `13 MB`
- `bench_disk.index`: about `196 MB`

### Measured Result

| Mode | deserialize_ms | avg_search_ms | rss_after_load_kb | rss_after_search_kb | vmhwm_kb |
|---|---:|---:|---:|---:|---:|
| `use_mmap=0` | 20.41 | 94.98 | 33108 | 48096 | 48096 |
| `use_mmap=1` | 11.88 | 99.94 | 20660 | 48080 | 48080 |

### Delta

Compared with `use_mmap=0`:

- `Deserialize` improved by about `42%`
- average search latency regressed by about `5%`
- RSS immediately after load dropped by about `12.4 MB`
- RSS after search was almost unchanged
- peak RSS was almost unchanged

## Interpretation

### What Improved

- mmap avoids copying `pq_compressed_vectors` into anonymous heap memory during load
- load latency improved noticeably
- load-time RSS reduction roughly matches the PQ compressed file size

### What Did Not Improve

- repeated search latency did not improve
- steady-state RSS after search did not improve much

### Why

Stage 1 only changed one file:

- `pq_compressed_vectors`

The following are still unchanged:

- `disk.index` access path
- medoids / centroids / norms loading
- graph traversal I/O path

Once search touches the mapped PQ pages, they still become resident, so RSS tends to converge with the non-mmap path.
The main gain in this stage is reducing copy cost and initial anonymous memory usage during load.

## Risks And Notes

- Search-time performance can be slightly worse due to page-fault / access-pattern differences.
- RSS alone does not fully describe reclaimability:
  - heap-loaded PQ data becomes anonymous memory
  - mmap-loaded PQ data remains file-backed
- Even when RSS converges, mmap-backed pages are still easier for the OS to reclaim than anonymous heap pages.

## Next Stage Template

Copy and append a new section like this for each stage:

```md
## Stage N Result

### Scope

- describe which files or paths were switched to mmap

### Validation

- unit tests:
- benchmark config:

### Measured Result

| Mode | deserialize_ms | avg_search_ms | rss_after_load_kb | rss_after_search_kb | vmhwm_kb |
|---|---:|---:|---:|---:|---:|
| use_mmap=0 |  |  |  |  |  |
| use_mmap=1 |  |  |  |  |  |

### Interpretation

- what improved
- what regressed
- why this happened
```

## Additional Scenario: Larger Dataset Search

This scenario was added after the Stage 1 baseline to compare mmap behavior on a larger index.
It is intended to reduce the chance that search-time behavior is dominated by a very small working set.

### Why This Case Was Added

There is no explicit "small dataset" threshold in the current `cached_beam_search()` implementation that forces brute-force search.
The brute-force fallback is triggered by:

- very high bitset filter ratio
- or `k_search > 0.5 * (num_points - bv_cnt)`

In this benchmark:

- no bitset filter is used
- `k = 10`
- `num_points = 200000`

So this is a normal search-path comparison on a larger index, not a brute-force fallback case.

### Benchmark Setup

Historical note:

- the larger random-data result in this section was collected with an ad-hoc runner during bring-up
- that temporary runner has been removed
- the formal repository entry for ongoing mmap evaluation is now `benchmark/run_diskann_mmap_eval.sh` with target `benchmark/diskann_mmap_eval`

Configuration:

- dataset size: `200,000 x 512` fp32
- metric: `L2`
- `k = 10`
- search queries: `200`
- search rounds: `10`
- `search_cache_budget_gb = 0`
- compare:
  - `use_mmap = 0`
  - `use_mmap = 1`

The current formal SIFT evaluation flow is:

```bash
benchmark/run_diskann_mmap_eval.sh
```

### Benchmark Files

Observed benchmark index file sizes:

- `bench_pq_compressed.bin`: about `49 MB`
- `bench_disk.index`: about `782 MB`

### Measured Result

| Mode | deserialize_ms | avg_search_ms | rss_after_load_kb | rss_after_search_kb | vmhwm_kb |
|---|---:|---:|---:|---:|---:|
| `use_mmap=0` | 60.21 | 115.78 | 70616 | 85592 | 85592 |
| `use_mmap=1` | 20.28 | 123.04 | 20644 | 80096 | 80096 |

### Delta

Compared with `use_mmap=0`:

- `Deserialize` improved by about `66%`
- average search latency regressed by about `6%`
- RSS immediately after load dropped by about `49.8 MB`
- RSS after search dropped by about `5.4 MB`
- peak RSS dropped by about `5.4 MB`

### Interpretation

Compared with the `50,000`-row baseline, the larger dataset keeps a clearer memory gap after repeated search:

- load-time memory benefit scales with the PQ compressed file size
- steady-state RSS no longer fully converges
- mmap still does not improve search latency in this stage

A likely explanation is that the larger graph search working set does not end up touching the full PQ compressed file as uniformly as the smaller baseline, so the mmap-backed PQ pages remain only partially resident.
This is still an inference from the measurements, not a direct page-fault trace.

### Environment Note

During the `use_mmap=1` run, the process logged repeated `io_setup() ... errno=11` messages when initializing AIO contexts.
The benchmark still completed successfully and produced results, so these were treated as environment-level warnings rather than benchmark failures.

## Open Dataset Validation

Using a public dataset is more convincing than purely random data because:

- recall can be checked against published ground truth
- vector distribution is more realistic
- query behavior is closer to standard ANN benchmarking practice

The repository already contains a suitable path for this via the HDF5 benchmark:

- dataset format: `ann-benchmarks` HDF5
- recommended first dataset: `sift-128-euclidean.hdf5`
- existing benchmark entry: [`benchmark/hdf5/benchmark_float.cpp`](/workspace/milvus/knowhere/benchmark/hdf5/benchmark_float.cpp)
- dataset preparation script: [`benchmark/prepare.sh`](/workspace/milvus/knowhere/benchmark/prepare.sh)

### Why SIFT1M First

`SIFT1M` is a good first public validation dataset for DiskANN mmap because:

- it is already referenced by the repository benchmark code
- metric is `L2`, which matches the simplest current DiskANN path
- it is large enough to make PQ and disk-index effects visible
- ground truth is included in the HDF5 file

### Current Benchmark Hook

The DiskANN path in [`benchmark/hdf5/benchmark_float.cpp`](/workspace/milvus/knowhere/benchmark/hdf5/benchmark_float.cpp) now runs both:

- `use_mmap = false`
- `use_mmap = true`

and prints:

- deserialize time
- search elapsed time
- recall against HDF5 ground truth

### Suggested Run Flow

1. Download the public dataset.
2. Build the benchmark target.
3. Run the DiskANN benchmark on `sift-128-euclidean.hdf5`.
4. Record recall, deserialize time, and search latency for both mmap modes in this document.

Example commands:

```bash
benchmark/prepare.sh
cmake --build build/Release --target benchmark_float -j2
cd output/unittest
LD_LIBRARY_PATH=/workspace/milvus/knowhere/build/Release:/workspace/milvus/knowhere/build/generators:${LD_LIBRARY_PATH:-} \
    /workspace/milvus/knowhere/build/Release/benchmark_float --gtest_filter=Benchmark_float.TEST_DISKANN
```

### Metrics To Record

For the public-dataset run, record at least:

- dataset name and file size
- `use_mmap`
- deserialize time
- search latency
- recall at `k`
- RSS after load if measured separately
- RSS after repeated search if measured separately

### Next Recommended Datasets

After `SIFT1M`, the next most useful datasets would be:

- `glove-100-angular`
  This is useful once IP/COSINE-related mmap behavior also matters.
- `nytimes-256-angular`
  This gives a different sparse-ish semantic distribution and a smaller footprint than GloVe.

## Open Dataset Result: SIFT Public Subset

A public-dataset validation was run using the `ann-benchmarks` SIFT HDF5 file.
To keep the turnaround reasonable while still using non-random data, the benchmark used a real-data subset rather than the full 1M-point train set.

### Scope

- source dataset: `sift-128-euclidean.hdf5`
- base subset: first `200,000` train vectors
- queries: first `1,000` test vectors
- metric: `L2`
- exact ground truth: recomputed on the `200,000`-vector subset with brute-force top-100
- index type: DiskANN
- changed mmap scope: only `pq_compressed_vectors`

### Benchmark Files

Observed benchmark file sizes:

- source HDF5: about `501 MB`
- subset base file: about `98 MB`
- subset query file: about `501 KB`
- subset GT ids file: about `391 KB`
- `sift_open_pq_compressed.bin`: about `13 MB`
- `sift_open_disk.index`: about `157 MB`

### Measured Result

| Mode | deserialize_ms | avg_search_ms | recall_at_100 | rss_after_load_kb | rss_after_search_kb | vmhwm_kb |
|---|---:|---:|---:|---:|---:|---:|
| `use_mmap=0` | 9.35 | 1360.55 | 0.99923 | 32736 | 46552 | 46552 |
| `use_mmap=1` | 9.04 | 949.94 | 0.99923 | 20276 | 46532 | 46532 |

### Delta

Compared with `use_mmap=0`:

- `Deserialize` improved by about `3%`
- average search latency improved by about `30%`
- `recall_at_100` was unchanged
- RSS immediately after load dropped by about `12.2 MB`
- RSS after search was effectively unchanged
- peak RSS was effectively unchanged

### Interpretation

This public-data result is more convincing than the earlier random-data runs because:

- the vectors come from a standard ANN benchmark dataset
- recall is checked against exact subset ground truth
- the search workload is less artificial than uniform random vectors

The main confirmed effect is still the same as Stage 1:

- mmap removes the eager heap copy of `pq_compressed_vectors`
- load-time RSS drops by about the PQ file size
- final steady-state RSS still converges because search touches much of the PQ working set

In this specific SIFT subset run, search latency was also better with mmap.
That is an observed result from this run, not yet a guaranteed trend.
It may be influenced by the real-data access pattern, page-cache state, or other environment effects, so it should be rechecked in repeated runs.

## Open Dataset Result: Full SIFT1M Benchmark

The earlier full-SIFT result that ran `use_mmap=0` and `use_mmap=1` back-to-back inside the same process is no longer considered valid for performance comparison.
That method allowed the first run to warm the OS page cache for both `l2_pq_compressed.bin` and `l2_disk.index`, which could bias the second run.

The result below replaces that earlier conclusion.
Each mode is now run in a separate process, and all generated DiskANN index files are explicitly evicted from page cache with `vmtouch -e` before each run.

### Scope

- source dataset: `sift-128-euclidean.hdf5`
- train vectors: `1,000,000`
- queries: `10,000`
- metric: `L2`
- topk: `100`
- index path: `build/Benchmark/benchmark/hdf5/diskann_test/l2_index/l2`
- changed mmap scope: only `pq_compressed_vectors`

### Benchmark Files

Observed DiskANN index file sizes from the full benchmark run:

- `raw_data`: about `489 MB`
- `l2_pq_compressed.bin`: about `62 MB`
- `l2_disk.index`: about `782 MB`
- `l2_sample_data.bin`: about `49 MB`

### Cache Control

Before each benchmark process:

- `vmtouch -e` was run on all `7` files in `diskann_test/l2_index`
- total evicted pages: `228,193`
- total evicted size: about `891 MB`
- verified post-eviction residency: `0/228193` pages resident, `0%`

### Measured Result

| Mode | deserialize_ms | search_ms | recall_at_100 | rss_after_load_kb | rss_after_search_kb | vmhwm_kb |
|---|---:|---:|---:|---:|---:|---:|
| `use_mmap=0` | 88.987 | 22243.6 | 0.997451 | 90760 | 120900 | 120900 |
| `use_mmap=1` | 13.5043 | 22505.3 | 0.997451 | 28304 | 120900 | 120900 |

### Delta

Compared with `use_mmap=0`:

- deserialize time is lower with mmap by about `75.5 ms` (`84.8%`)
- load-time RSS is lower with mmap by about `62.5 MB`
- search time is slightly worse with mmap by about `261.7 ms` (`1.2%`)
- `recall_at_100` is identical in this run
- post-search RSS and HWM converge to the same value

### Interpretation

This is the strongest result in the document so far because it uses:

- the full public `SIFT1M` train set
- all `10,000` public queries
- separate processes for the two modes
- explicit page-cache eviction before each run

The corrected conclusion is different from the earlier same-process benchmark:

- Stage 1 mmap clearly reduces deserialize latency
- Stage 1 mmap clearly reduces load-time RSS by approximately the PQ file size
- Stage 1 mmap does not show a meaningful search-speed improvement once page-cache contamination is removed
- search-phase RSS still converges because the workload touches the PQ working set during search

The recall result remains unchanged, which supports that the current Stage 1 mmap change does not materially alter result quality.

### Notes

- The earlier same-process full-SIFT result is intentionally superseded by this section and should not be used for performance conclusions.
- This corrected run used a dedicated temporary benchmark runner so RSS and HWM could be recorded in addition to latency and recall.

## Formal Benchmark Run: 2026-03-22 08:36:06 UTC

### Scope

- entry script: benchmark/run_diskann_mmap_eval.sh
- benchmark target: benchmark/diskann_mmap_eval
- dataset: /tmp/open_dataset_bench/sift-128-euclidean.hdf5
- work directory: /workspace/milvus/knowhere/build/Benchmark/benchmark/hdf5/diskann_test
- index directory: /workspace/milvus/knowhere/build/Benchmark/benchmark/hdf5/diskann_test/l2_index

### Cache Control

- use_mmap=0 residency after eviction: Resident Pages: 0/228193  0/891M  0%
- use_mmap=1 residency after eviction: Resident Pages: 0/228193  0/891M  0%

### Measured Result

| Mode | deserialize_ms | search_ms | recall_at_100 | rss_after_load_kb | rss_after_search_kb | vmhwm_kb |
|---|---:|---:|---:|---:|---:|---:|
| use_mmap=0 | 120.318 | 23502.2 | 0.997451 | 103488 | 136292 | 136292 |
| use_mmap=1 | 22.0772 | 22689.3 | 0.997451 | 41032 | 136592 | 136592 |
