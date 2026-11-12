# Baseline versus exact Uber PR and final partition stack

## Acceptance criterion

Gate the completed optimization against the original baseline. Small incremental differences between adjacent split PRs are informational when the completed series improves that same workload versus baseline. Intermediate improvements are not required to be monotonic on every fixture.

## Sources and reuse

- Baseline: `65b9b3c222ee7d6ba8f0f442688bd64268709e93`.
- Exact, unmodified NVIDIA/cudf#25154 snapshot: `c285951eeddfae0a34379d82f0c40dbbd264e695`.
- Final guarded stack (#9 through #18): `74fe237ab462746119e0e9bb82e5f152ac31d3d5`.

The source snapshot is pinned; this report does not claim to have refreshed the upstream head. All eight original partition-source fingerprints, including the unchanged round-robin file, match the local Git snapshot. The baseline source and both historical native-library fingerprints also match. Native libraries and compatible JNI build results are reused without cleaning or overwriting existing work. The final stack is the exact published fork head.

## Main multi-column STRING result

All **26** regular large multi-column STRING fixtures in the fresh matrix improved:

| Comparison versus baseline | Fixtures | Geometric mean wall speedup | Minimum observed speedup |
| --- | ---: | ---: | ---: |
| Exact Uber | 26 | **2.207x** | 1.497x |
| Final stack | 26 | **2.201x** | 1.496x |

The separate historical regular multi-column STRING cohort has **98 fixtures at 2.161x** for exact Uber versus baseline. These are distinct cohorts with some overlap and are not multiplied or pooled into a single speedup.

## Whole matrix and the confirmed exception

| Fresh screening cohort | Fixtures | Exact Uber / baseline | Final stack / baseline |
| --- | ---: | ---: | ---: |
| All fresh fixtures | 210 | 2.856x | 2.939x |
| Partially empty STRING columns | 14 | 2.351x | 2.595x |
| Partially all-null columns | 16 | 2.791x | 2.570x |

The exact Uber snapshot is **not faster on every tested fixture**. Two nullable STRUCT columns, 1,025 rows and one partition reproduced a slowdown. The current stack retains standard table/column copies for this identity schema:

| STRUCT identity copy | Paired geometric mean speedup | 95% interval | Median-of-run-medians ratio |
| --- | ---: | ---: | ---: |
| Exact Uber / baseline | **0.738x** | 0.703–0.775x | 0.750x |
| Final stack / baseline | **2.235x** | 2.187–2.284x | 2.257x |

No nominal slowdown was established for the final stack in the six-block confirmation of all ten negative screening candidates. Its worst median-of-run-medians ratio was 0.994x. Some intervals remain broad, so this is evidence of no reproducible regression in the tested matrix, not proof of literally zero slowdown on every possible workload. Unoptimized controls can remain flat.

## The small incremental #11 case

The 1,025-row, three-STRING-column, 256-partition fixture that showed a small loss versus #10 has the following complete-series result:

| Complete solution / baseline | Observed wall speedup |
| --- | ---: |
| Exact Uber | 2.762x |
| Final stack | 3.306x |

The incremental loss does not erase the preceding STRING materialization gains. No additional small-input heuristic or exploratory one-tile/legacy fallback PoC is adopted into #11 on that basis. Keep the validated host guards that protect the final baseline-relative behavior.

## Method and scope

NVIDIA L4, CUDA 12.9.1/GCC 14, Release O3, sm89, per-thread default stream. A fresh common executable verifies output partition boundaries, schema and per-partition row-hash multisets for every variant. The 210-case screen uses the same executable and fixture file for all three libraries, at least 150 ms warmup, 31 samples, default GPU clocks and CPU affinity 0,2. Input construction and verification are excluded; output destruction is included. Per-case screening speedup is baseline median wall time / candidate median wall time; cohort values are geometric means. Original screening values are retained in aggregates rather than selectively replaced with reruns.

Ten fixtures had a wall-speedup point estimate below 1/1.02 in at least one candidate. They were repeated in all six permutations of baseline/Uber/stack order, with 301 samples per fixture per independent process, at least 500 ms warmup and SM clocks configured at 2040 MHz. GPU use is exclusive, compilation is paused, loaded-library/executable/fixture fingerprints are checked, and clock defaults are restored afterward. The follow-up uses paired log ratios across the six independent process blocks for Student-t 95% intervals; within-process samples are not independent repetitions. The secondary median ratio is median(baseline run medians) / median(candidate run medians). Per-fixture intervals are not adjusted for multiple comparisons. An interval crossing 1 does not establish absence of a regression.

The historical 348-case matrix used 15 samples per fixture and 150 ms warmup, with 15 selected candidates rechecked in three alternating-order repeats of 30 samples. Its overall speedup was 3.053x, with no confirmed >5% regression under that earlier screen. The present matrix adds exact 1,024/1M/4M boundaries, LIST/STRUCT identity copies, INT64 STRING offsets, lowered output-offset thresholds, partially empty/null columns, and slices. The historical claim is not a guarantee that every scenario is faster; the new STRUCT counterexample demonstrates why the extra coverage matters.

## Every negative screening candidate

| Fixture | Solution | Initial speedup | Paired follow-up | Median ratio | 95% interval |
| --- | --- | ---: | ---: | ---: | ---: |
| `strings-1023-2-1-8-32-0-0-0-7` | uber | 0.932x | 0.960x | 0.994x | 0.841–1.097x |
| `strings-1023-2-1-8-32-0-0-0-7` | stack | 0.972x | 0.957x | 1.006x | 0.844–1.085x |
| `strings-1023-2-2-8-32-0-0-0-7` | uber | 0.616x | 1.048x | 1.002x | 0.777–1.413x |
| `strings-1023-2-2-8-32-0-0-0-7` | stack | 0.997x | 1.119x | 1.027x | 0.903–1.386x |
| `strings-1000003-257-4-8-32-0-0-0-0` | uber | 0.983x | 1.004x | 0.996x | 0.983–1.026x |
| `strings-1000003-257-4-8-32-0-0-0-0` | stack | 0.930x | 1.003x | 1.003x | 0.977–1.030x |
| `fixed-1023-3-1-16-32-0-0-0-0` | uber | 0.945x | 0.943x | 0.982x | 0.811–1.095x |
| `fixed-1023-3-1-16-32-0-0-0-0` | stack | 0.615x | 0.993x | 1.000x | 0.957–1.030x |
| `fixed-1024-3-1-16-32-0-0-0-0` | uber | 1.161x | 1.243x | 1.264x | 1.187–1.301x |
| `fixed-1024-3-1-16-32-0-0-0-0` | stack | 0.900x | 1.241x | 1.261x | 1.196–1.288x |
| `fixed-1025-256-1-16-32-0-0-0-0` | uber | 1.167x | 1.270x | 1.253x | 1.147–1.406x |
| `fixed-1025-256-1-16-32-0-0-0-0` | stack | 0.859x | 1.250x | 1.242x | 1.176–1.329x |
| `fixed-1000000-257-1-16-32-0-0-0-0` | uber | 0.989x | 0.977x | 0.998x | 0.942–1.014x |
| `fixed-1000000-257-1-16-32-0-0-0-0` | stack | 0.946x | 0.986x | 0.994x | 0.960–1.013x |
| `strings-n1000000-p257-c4-l32-z0-d2-s0` | uber | 0.976x | 0.995x | 0.995x | 0.984–1.005x |
| `strings-n1000000-p257-c4-l32-z0-d2-s0` | stack | 0.972x | 0.998x | 1.000x | 0.987–1.008x |
| `identity-10` | uber | 0.770x | 0.738x | 0.750x | 0.703–0.775x |
| `identity-10` | stack | 2.279x | 2.235x | 2.257x | 2.187–2.284x |
| `fixed-n1024-p257-c2-w1-l0-z0-m0-d0-s0` | uber | 0.989x | 1.094x | 1.016x | 0.900–1.331x |
| `fixed-n1024-p257-c2-w1-l0-z0-m0-d0-s0` | stack | 0.672x | 1.088x | 1.017x | 0.879–1.347x |

Matrices, exact source/library manifests, screening samples and every follow-up sample are archived alongside this report. All timing attempts included in the report passed output verification and GPU-isolation checks. The final guarded prefix also passed 1,900 enabled partition tests, seven stream tests and focused memcheck/synccheck. Prior dirty JNI source files and original build caches are preserved.
