# Partition optimization stack: performance reports

Pinned source: NVIDIA/cudf#25154 at `c285951eeddfae0a34379d82f0c40dbbd264e695`.
Baseline: `65b9b3c222ee7d6ba8f0f442688bd64268709e93`.
Hardware/toolchain: NVIDIA L4, CUDA 12.9.1, GCC 14, Release O3, sm89, per-thread default stream.

Every draft targets the immediately preceding branch. Production changed-line counts include additions and deletions and exclude tests. Each report compares its exact library prefix with its direct parent. The individual cohort speedups cannot be multiplied across different cohorts.

| Stage | Draft PR | Production lines | Report |
| --- | --- | ---: | --- |
| 1: Centralize partition policy and fast-path trivial inputs | [9](https://github.com/res-life/cudf/pull/9) | 230 | [report](reports/pr-01.md) |
| 2: Batch multi-column STRING materialization in partition | [10](https://github.com/res-life/cudf/pull/10) | 297 | [report](reports/pr-02.md) |
| 3: Build multi-column STRING gather maps with binary and tile ranks | [11](https://github.com/res-life/cudf/pull/11) | 286 | [report](reports/pr-03.md) |
| 4: Fuse binary STRING sizes and pack large multi-column ranks | [12](https://github.com/res-life/cudf/pull/12) | 299 | [report](reports/pr-04.md) |
| 5: Materialize fixed-width partition output directly | [13](https://github.com/res-life/cudf/pull/13) | 285 | [report](reports/pr-05.md) |
| 6: Reorder fixed-width writes using the existing partition policy | [14](https://github.com/res-life/cudf/pull/14) | 56 | [report](reports/pr-06.md) |
| 7: Recompute large single-column STRING targets without a row map | [15](https://github.com/res-life/cudf/pull/15) | 103 | [report](reports/pr-07.md) |
| 8: Use tile-rank gather fallbacks for remaining partition schemas | [16](https://github.com/res-life/cudf/pull/16) | 221 | [report](reports/pr-08.md) |
| 9: Scatter single fixed-width fallback output with saved tile ranks | [17](https://github.com/res-life/cudf/pull/17) | 165 | [report](reports/pr-09.md) |
| 10: Batch identity STRING copies and omit uniform columns | [18](https://github.com/res-life/cudf/pull/18) | 161 | [report](reports/pr-10.md) |

## Reproduction

The native harness uses cuDF test column wrappers and should be compiled with the include/link settings of the matching cuDF native test build. Keep the same toolchain and architecture for all compared variants. `partition_bench_uniform.cpp` additionally creates partially uniform schemas for the identity/uniform-column report.

Run on an otherwise idle GPU, selecting the exact variant library:

```sh
LD_LIBRARY_PATH=<variant-library-directory>:$LD_LIBRARY_PATH ./partition_bench matrices/core.tsv 31
LD_LIBRARY_PATH=<variant-library-directory>:$LD_LIBRARY_PATH ./partition_bench_uniform matrices/identity.tsv 51
```

TSV columns: id, kind, rows, partitions, columns, fixed-width bytes, maximum string length, null percentage, force-mask flag, map pattern, slice offset. Map patterns: 0 uniform, 1 all-hot, 2 90%-hot, 3 tile-clustered. Timing is preceded by at least 150 ms warmup. A sample count of zero performs verification only. Input construction and row verification are excluded from timing; output destruction is included. `strings_limit` temporarily lowers the output-offset threshold; `strings64` provides INT64-offset input. `strings_uniform` and `fixed_uniform` make the first column all null and retain valid data in the remaining columns.

The harness prints the loaded library path and verifies partition boundaries, row/column counts, and per-partition row-hash multisets. The published provenance contains hashes and successful GPU-isolation checks; machine-local paths and process IDs are omitted. The accumulated partition suite contains 1,900 enabled tests and one pre-existing disabled test; stream validation contains seven tests. Memcheck and synccheck cover the representative optimized and fallback paths.

## Scope and deliberate guards

Core GPU kernels are extracted from the pinned source. STRING allocation rejection reuses the existing gather permutation, and configured lower output-offset thresholds retain the preceding small-schema path. Identity batching is restricted to multiple non-null STRING columns; standard copies are retained for other schemas. These guards address measured regressions in the original fallback flows. Reports disclose timing noise and repeated controls; they do not claim universal zero regression on unmeasured hardware or distributions.

## End-to-end baseline acceptance

[Baseline versus exact Uber and the final guarded stack: historical 348-case evidence, fresh 210-case three-way comparison, multi-column STRING gains, and confirmed STRUCT exception](uber-comparison/README.md). Small incremental split differences are informational when the completed series improves the same fixture versus baseline. Licensed native harnesses are provided with byte-exact compilation inputs in matching .txt files; source-input-map.json records both hashes.

## Regression follow-up

[Controlled per-PR checks, rejected/final #18 empty-column measurements, A/A calibration, and exact source manifests](regression-check/README.md). Final #18 retains valid empty STRING columns in partially uniform tables after measured route-downgrade regressions. All-null omission and identity batching remain enabled. Historical original #18 data remains archived; use the guarded-prefix results in its current report.
