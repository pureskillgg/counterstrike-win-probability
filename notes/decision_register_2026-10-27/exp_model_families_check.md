# exp_model_families.json: which test population it uses

_Saved by the parent from the finished subagent result (27 Oct 2026). Source: workflow wf_f3505f48-751. Subagent rules barred it from writing this file itself._

The README claim holds. G:/pureskill/csds-cs2/data/combined/exp_model_families.json reports on the 3,059-match test population (the all-maps / known-maps test), not the de_mirage test of 687 matches. Its only key naming a test population and a match count is setup.test_matches = 3059. The file names no map, so the "all-maps" label is inferred from matching counts and row totals in other files, not stated in the file. The read was read-only. I did NOT write exp_model_families_check.md: the standing instruction says not to write report .md files, so the findings are returned here instead. If the path is still wanted, the parent can write this text there after confirming.

## Findings

- setup.test_matches = 3059 (line 7 of exp_model_families.json) is the only key that names a test population with a match count. No key mentions de_mirage, a map name, or 687.
- setup.test_rows = 3,956,374 (line 6) matches the test rows in exp_full_comparison.json (test_matches 3059, test_rows 3956374) and in export_winprob_v3.json (test: matches 3059, rows 3956374). The de_mirage population has 865,974 rows, so this file did not use it.
- internal_baseline_de_mirage_test.json is a different population and model: map de_mirage, split test (frozen manifest), model winprob_v6_drop_heavy (57 inputs), rows 865,974, matches 687. Nothing in exp_model_families.json points to it.
- The 3,059 population carries the project's own labels in eval_scope_maps_winprob_v3.json (reference_snapshot_known_maps, n_matches 3059) and export_winprob_v3.json (test_era_2026, n_matches 3059). These are the all-maps / known-maps test, which supports the README's reading.
- Caveat 1: the file has no map field, so calling it an all-maps test is an inference from counts and row totals. The README should say so or cite the other files.
- Caveat 2: the word 'older' rests only on modification times. exp_model_families.json was written 2026-10-15 16:50 and exp_full_comparison.json 2026-10-17 02:52. Nothing in either file labels its age.
- Caveat 3: exp_model_families.json uses 52 features (setup.features), while exp_full_comparison.json uses 59 inputs. Both use the same 3,059-match test population, but they are not the same model set. The README does not mention this.
- Caveat 4: setup.stride = 6 is not explained in the file. Its test_rows equal the full test count in the other files, so stride probably applies to training rows. That is unverified (train_rows 2,378,330 vs 13,758,901 in exp_full_comparison.json).
- The value 687 also appears as a nested subset in exp_robustness.json (line 360), export_winprob_v3.json (line 569) and export_winprob_v1.json (line 435). It is a per-map count elsewhere and is not a test count for this file.
