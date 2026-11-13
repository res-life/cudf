# Independent L4 recheck, 2026-11-13

The single-column 16-byte case retains a slowdown signal, but its size is unstable. The two-column INT64 case is inconclusive. This recheck does **not** establish a less-than-2% regression bound for either case. It neither pools with nor invalidates the earlier 384-process experiment on its original host.

Exact PR head: `250fdcc6b207542652807251ef5ff994110ba711`; direct parent: `39b6dcf0ea66a53b0721341bfefe0f5356286c6f`.

## Primary results

Candidate/parent wall-time ratios; values above 1 mean slower. Intervals are individual, two-sided 95% Student-t intervals across 16 quartet log ratios, not across the 10,001 iterations inside each process. Slowdown p-values are one-sided. Holm2 covers the two targets prespecified for this recheck; it must not be interpreted as the earlier Holm42 family.

| Fixture | Ratio | Individual 95% CI | Student-t Holm2 p | Exact randomization Holm2 p |
| --- | ---: | --- | ---: | ---: |
| `fixed-1000000-2-1-16-32-0-0-0-0` | 1.03851 (+3.85%) | 1.00364–1.07460 | 0.03236 | 0.04009 |
| `fixed-1000000-2-2-8-32-0-0-2-0` | 1.01515 (+1.51%) | 0.99000–1.04094 | 0.11041 | 0.11033 |

Conservative original-family-42 sensitivity gives Student-t adjusted p=0.67963/1.0 and randomization adjusted p=0.84196/1.0. The single-column result is significant within this targeted two-case recheck, not under that conservative 42-case sensitivity. Its interval includes changes below and above 2%; it does not establish a regression exceeding 2% either.

## Design and environment

- NVIDIA L4, CUDA 12.9.1 JNI container, existing compatible JNI build cache, separate relinked parent/candidate libraries. Only the partitioning translation unit differs; all other partition sources are byte-identical. Library loading and hashes were verified for every process.
- 16 primary quartets: 8 ABBA and 8 BAAB, shuffled before measurement with seed 20261113. Each process runs both target fixtures with 1 second of warmup per fixture and 10,001 measured iterations. All 64 primary processes were retained.
- Separately, 4 diagnostic control quartets (16 processes) run the targets plus single-column 8-byte/P2 and single-column 16-byte/P3 neighbors. Total: 80 independent processes. No outcome-dependent extension or exclusion; campaigns are not pooled.
- Benchmark affinity 0,2 (one physical core); monitor 1,3 (the other physical core). The earlier report used benchmark 2,3 and monitor 0,8, so this is a different host CPU topology. CPU frequency was not controlled.
- GPU lock requests succeeded: SM 2040 MHz, memory 6251 MHz. All observations inside timed windows matched these clocks. The approximately 1 Hz monitor covered 173/192 fixture timing windows; 19 windows had no clock sample. GPU process isolation was checked throughout. Default clocks were restored afterward.
- Exact assignment randomization enumerates all 12,870 balanced assignments of 8 ABBA/8 BAAB quartet orientations. The analysis unit is the independent quartet, not the within-process iteration.

## Diagnostics and interpretation

The separate four-quartet controls yielded ratios 1.00837 [0.97768,1.04002] and 1.04159 [0.92420,1.17388] for the two targets. Neighbor controls were 1.02396 [0.98244,1.06725] (8-byte/P2) and 0.99979 [0.90763,1.10131] (16-byte/P3). These broad intervals do not independently establish a stable slowdown magnitude. Some same-version process pairs also varied substantially; no samples were removed.

Source inspection confirms both targets retain the same direct-output plan and return before the new fallback. All 537 common `partition_map_plan`/`partition_fixed` GPU functions have identical complete per-function SASS text in the sm89 partitioning objects, including the 16-byte single-output and nonnullable table-output recompute specializations. This rules out a changed instruction sequence in those compared kernels; it does not rule out host dispatch, binary layout, or process-level effects. The cause of the measured signal remains unresolved.

The earlier report's +1.24%/+1.33% estimates and individual upper bounds below 2% describe that experiment and host. A significant slowdown and an upper bound below 2% are compatible statements. This independent run cannot generalize that bound to all L4 systems. No production code or heuristic was changed as part of this recheck.

## Reproduction artifacts

`summary.json`, `preregistered-plan.json`, fixtures, source/library/executable hashes, `sass-audit.json`, and `telemetry-summary.json` accompany this report. `measurements.tar.gz` contains all raw per-iteration records, sanitized per-process provenance, and clock/process-isolation observations. Unpack it in this directory and run `python3 analyze.py`. `partition_bench_review.cpp.txt` is the byte-exact measured benchmark source; `benchmark-build.json` records its source/executable hashes. The `.txt` suffix preserves the measured source without an editorial header change.
