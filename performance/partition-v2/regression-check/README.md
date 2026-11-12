# Per-PR regression follow-up

All comparisons are against each draft's exact direct parent. The initial survey covers 123 fixtures across 10 PRs and uses six independent balanced AB/BA blocks, 301 samples per fixture per process, 500 ms warmup, a 2040-MHz SM setting and benchmark CPU 0. Every unresolved interval or nominal negative reading is independently checked using CPUs 0,2 on the same physical core; these measurements are reported separately. Monitoring uses CPUs 1,3. Default GPU clocks were restored after timing. No compilation runs during measurements.

## Confirmed and addressed regression

Additional partial-empty fixtures found a real route downgrade in the original #18 extraction: two-column STRING tables with a valid empty column measured about 0.81x, 0.90x and 0.75x versus the direct parent. The final host guard retains valid empty columns in partially uniform tables. All-null omission and multi-column STRING identity batching remain enabled. The rejected two-block measurements and final six-block 24-fixture matrix are both retained. The final prefix passed 1,900 enabled partition tests, seven stream tests, memcheck and synccheck. Original cached binaries and dirty JNI files were preserved.

## Independent confirmation

The following table reports per-fixture Student-t 95% intervals on paired log ratios across six independent process blocks. A separately declared 2% noninferiority bound needs a lower speedup bound of at least 1/1.02. Literal nominal slowdowns are reported even when smaller than 2%. An interval containing 1 does not prove absence of a regression. Within-process timing samples are not independent replications; intervals are not adjusted for multiple comparisons. A/A controls use identical library bytes and are not optimization speedups.

| PR | Confirmation fixtures | Nominal slowdowns | Established >2% slowdowns | Unresolved 2% bounds |
| --- | ---: | ---: | ---: | ---: |
| [#9](https://github.com/res-life/cudf/pull/9) | 1 | 0 | 0 | 1 |
| [#10](https://github.com/res-life/cudf/pull/10) | 3 | 0 | 0 | 3 |
| [#11](https://github.com/res-life/cudf/pull/11) | 5 | 1 | 0 | 4 |
| [#12](https://github.com/res-life/cudf/pull/12) | 5 | 0 | 0 | 4 |
| [#13](https://github.com/res-life/cudf/pull/13) | 1 | 0 | 0 | 1 |
| [#14](https://github.com/res-life/cudf/pull/14) | 11 | 0 | 0 | 10 |
| [#15](https://github.com/res-life/cudf/pull/15) | 15 | 0 | 0 | 13 |
| [#16](https://github.com/res-life/cudf/pull/16) | 9 | 0 | 0 | 8 |
| [#17](https://github.com/res-life/cudf/pull/17) | 9 | 0 | 0 | 4 |
| [#18](https://github.com/res-life/cudf/pull/18) | 35 | 1 | 0 | 31 |

See each PR description and summary JSON for the complete fixture-level outcome. The initial #18 survey used the pre-guard head in stages-before-guard.json; it is not a final-prefix measurement. Fresh default-clock core/identity results for final #18 are guard-core/identity-comparison.json; all final uniform-boundary and independent confirmation data use the final guarded prefix. Other production heads are unchanged.

The native harness uses the matching cuDF native test build include/link settings. Run with the exact variant selected through LD_LIBRARY_PATH and preserve the same hardware/toolchain. TSV fields and workload generation are described in the parent README. The partial-empty harness checks the actual metadata admission condition before timing. Hashes, successful output checks, and GPU isolation are in provenance.json. Machine-local paths and process IDs are omitted from published provenance.
