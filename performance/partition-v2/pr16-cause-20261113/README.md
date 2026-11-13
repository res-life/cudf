# PR16 targeted repeat and same-library controls, 2026-11-13

The previous slowdown did not reproduce in this independent repeat. Both point estimates are now faster, but the individual 95% intervals still include parity and slowdowns above 2%. This does not establish a universal no-regression bound or a specific root cause. Previous experiments remain retained in the PR description; results are not pooled or substituted selectively.

Exact candidate: `250fdcc6b207542652807251ef5ff994110ba711`. Exact direct parent: `39b6dcf0ea66a53b0721341bfefe0f5356286c6f`. Same libraries and uninstrumented executable as the preceding independent L4 recheck, with loading and hashes verified for every process.

## Uninstrumented primary repeat

12 prespecified balanced shuffled ABBA/BAAB quartets (48 independent processes), 1 second warmup per fixture, 10,001 iterations per fixture/process. No exclusions or outcome-dependent extension. Ratios are candidate/parent; above 1 is slower. Individual two-sided 95% Student-t intervals use quartet log ratios, not the iterations as independent samples. One-sided slowdown p-values use Holm over these two prespecified targets.

| Fixture | Ratio | Individual 95% CI | One-sided Student-t Holm2 p |
| --- | ---: | --- | ---: |
| `fixed-1000000-2-1-16-32-0-0-0-0` | 0.99579 (-0.42%) | 0.96069–1.03218 | 1.00000 |
| `fixed-1000000-2-2-8-32-0-0-2-0` | 0.97262 (-2.74%) | 0.92310–1.02480 | 1.00000 |

## Same-library A/A controls

Six additional balanced randomized quartets for each library separately, 24 processes for parent and 24 for candidate. Nominal A and B use byte-identical library files within each campaign. The ratio is nominal B/A, not candidate/parent; it measures process variation without any code change. These campaigns are analyzed separately and not pooled with the primary comparison. Intervals are individual 95% quartet intervals.

| Campaign | Fixture | Nominal B/A | Individual 95% CI |
| --- | --- | ---: | --- |
| aa-parent | `fixed-1000000-2-1-16-32-0-0-0-0` | 0.99797 | 0.91762–1.08535 |
| aa-parent | `fixed-1000000-2-2-8-32-0-0-2-0` | 0.99783 | 0.96516–1.03161 |
| aa-candidate | `fixed-1000000-2-1-16-32-0-0-0-0` | 1.01907 | 0.96227–1.07923 |
| aa-candidate | `fixed-1000000-2-2-8-32-0-0-2-0` | 0.99380 | 0.97618–1.01174 |

Same-version process-pair ranges (diagnostic extremes, not overall speedups):

- aa-parent, `fixed-1000000-2-1-16-32-0-0-0-0`: later/earlier ratios 0.85659–1.44009 across all 12 same-version process pairs.
- aa-parent, `fixed-1000000-2-2-8-32-0-0-2-0`: later/earlier ratios 0.85287–1.05267 across all 12 same-version process pairs.
- aa-candidate, `fixed-1000000-2-1-16-32-0-0-0-0`: later/earlier ratios 0.96842–1.34922 across all 12 same-version process pairs.
- aa-candidate, `fixed-1000000-2-2-8-32-0-0-2-0`: later/earlier ratios 0.94621–1.02585 across all 12 same-version process pairs.

Total uninstrumented experiment: 96 independent accepted processes. Raw process medians, within-process samples and same-version process-pair ratios are retained. A/A variation characterizes uncertainty; it does not by itself prove that a candidate/parent difference is harmless.

## Actual GPU launch diagnostics

Eight separate Nsight Compute runs: parent/candidate/candidate/parent for each target, the first 100 launches per run, without a kernel-name filter. These prefixes cover initial output verification and warmup calls. SM/memory locks were requested at 2040/6251 MHz. Kernel replay uses `gpu__time_duration.sum` and `--cache-control none`. These profiler runs are excluded from the uninstrumented performance statistics.

All four profiles per target have identical ordered 100-launch prefixes, including count_tiles, scan initialization, scan, get_offsets, scatter_recompute and verification kernels. Matching grid/block geometry was also checked across the four profiles per target. Neither target launches the new fallback kernels. The 16-byte target uses single_output<16> and the INT64 target the nonnullable table_output sink. Matching per-kernel medians below average the two fresh processes per library; they are diagnostic estimates, not confidence-bounded speedups.

| Target index | Kernel | Candidate/parent kernel time |
| --- | --- | ---: |
| 0 | `void partition_map_plan::count_tiles<1, 0, 0, int>` | 1.00266 |
| 0 | `void scan::DeviceScanInitKernel<scan::policy_selector_from_types<int *, int *, int, unsigned int, std::plus<void>, 0>, int *, int *, cub::ScanTileState<int, 1>, int>` | 1.00000 |
| 0 | `void scan::DeviceScanKernel<scan::policy_selector_from_types<int *, int *, int, unsigned int, std::plus<void>, 0>, int *, int *, cub::ScanTileState<int, 1>, std::plus<void>, cub::InputValue<int, int *>, unsigned int, int, 0, 0, int>` | 0.99405 |
| 0 | `partition_map_plan::get_offsets` | 1.00000 |
| 0 | `void partition_map_plan::scatter_recompute<1, 1, partition_fixed::single_output<16>, int, int *>` | 0.99746 |
| 0 | `void for_each::static_kernel<for_each::policy_selector, int, hashing::xxhash_64` | 0.99760 |
| 1 | `void partition_map_plan::count_tiles<1, 0, 0, int>` | 1.00000 |
| 1 | `void scan::DeviceScanInitKernel<scan::policy_selector_from_types<int *, int *, int, unsigned int, std::plus<void>, 0>, int *, int *, cub::ScanTileState<int, 1>, int>` | 1.00000 |
| 1 | `void scan::DeviceScanKernel<scan::policy_selector_from_types<int *, int *, int, unsigned int, std::plus<void>, 0>, int *, int *, cub::ScanTileState<int, 1>, std::plus<void>, cub::InputValue<int, int *>, unsigned int, int, 0, 0, int>` | 0.99405 |
| 1 | `partition_map_plan::get_offsets` | 1.00000 |
| 1 | `void partition_map_plan::scatter_recompute<1, 0, partition_fixed::table_output<0>, int, int *>` | 0.99939 |
| 1 | `void for_each::static_kernel<for_each::policy_selector, int, hashing::xxhash_64` | 0.99417 |

These observations complement the preceding source/SASS audit: both plans return on the existing direct-output path, and all 537 common related GPU functions have identical SASS text. Equal source paths and instruction sequences do not force equal whole-call wall times: the CUDA-event interval also includes host launch/synchronization gaps, and separate processes can differ in scheduling, CPU frequency, memory placement and binary/driver state. Those are possible mechanisms, not established causes here. The full binary differs because PR16 adds other kernels and host dispatch code. This experiment does not isolate which mechanism produced the earlier signal.

The original CUPTI activity-reader attempt failed with CUPTI_ERROR_INVALID_KIND; an initial profiler configuration captured no kernels. Those diagnostic attempts were not used as timing evidence. The corrected Nsight profiles and all uninstrumented observations are retained separately.

## Controls and reproduction

NVIDIA L4, CUDA 12.9.1, JNI build-cache validation; benchmark affinity 0,2 and monitoring 1,3. CPU frequency was not controlled. GPU process isolation was checked per process; output verification, library/executable/fixture hashes and exact source revisions were checked. The timing-window telemetry summary records actual clock observation coverage rather than assuming every iteration was observed. Default GPU clocks were restored, and production code and shared JNI outputs were unchanged.

Unpack `measurements.tar.gz` here and run `python3 analyze.py`; its `summary-live.json` must equal `summary.json`. The archive contains all 96 uninstrumented per-iteration records, sanitized provenance and telemetry. Nsight CSV records, `profile-summary.json`, original source/library manifests and exact measured benchmark hashes accompany the report.
