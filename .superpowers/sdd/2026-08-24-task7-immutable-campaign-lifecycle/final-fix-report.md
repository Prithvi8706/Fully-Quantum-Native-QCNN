# Task 7 Final Fix Report

## Status

**DONE** — both Important blockers from `final-review.md` were corrected without changing campaign artifact formats, model semantics, the frozen FQCNN architecture, stale evidence, `.gitignore`, or `.claude/settings.local.json`.

## Fixes

### Queue release contention

- Queue acquisition remains non-blocking and exclusive.
- Owner release now waits for a compliant operation already holding the operation guard to finish, without deleting or breaking a stale guard automatically.
- After serialization, release still compares the exact owner and removes only that lease.
- Campaign launch now raises a cleanup error when owner release returns `False` instead of silently reporting success.
- A deterministic threaded regression holds the guard inside a foreign compliant acquire while the current owner releases, then proves the owner lease is removed and a later replacement lease cannot be deleted by the former owner.
- Prior foreign-owner and replacement-safety regressions remain passing.

### Exact seed and numeric config identity

- Reusable status evidence now requires `type(status["seed"]) is int`.
- A supplied expected seed now requires `type(seed) is int` and exact equality, with no coercion.
- Numeric values in config identity comparisons require matching exact numeric types, preventing boolean/integer and integer/float equality aliases while leaving artifact formats unchanged.
- Regressions cover recorded `false`, `0.5`, `true`, and `1.9` aliases for expected seeds `0` and `1`, non-integer expected seeds, and boolean aliases for numeric config identity.

## Verification

### Focused queue contention and replacement safety

```text
python -m pytest tests/test_campaign_manifest.py -k "owner_release_waits_for_compliant_acquire_contention or launch_propagates_owner_release_failure or release_cannot_delete_reacquired_lease or foreign_owner_cannot_release_live_lease or queue_acquisition_is_exclusive or queue_metadata_failure_rolls_back_lease" -q
```

Result: **6 passed, 205 deselected, 0 failed**.

### Focused malformed seed and config identity

```text
python -m pytest tests/test_resume_and_parallelism.py -k "boolean_and_fractional_recorded_seeds or non_integer_expected_seeds or boolean_numeric_config_aliases or malformed_status_identity or different_seed or different_config or complete_matching_run" -q
```

Result: **15 passed, 23 deselected, 0 failed**.

### Complete campaign lifecycle suite

```text
python -m pytest tests/test_campaign_manifest.py -q
```

Result: **209 passed, 2 skipped, 0 failed** in 6.97s.

Both skips are documented Windows `WinError 1314` symlink-privilege limitations.

### Complete resume and parallelism suite

```text
python -m pytest tests/test_resume_and_parallelism.py -q
```

Result: **36 passed, 2 skipped, 0 failed** in 6.48s.

Both skips are documented absent-MNIST conditions.

### Syntax and hygiene

```text
python -m py_compile QCNN/utils/exclusive_queue.py QCNN/utils/run_artifacts.py experiments/campaign.py tests/test_campaign_manifest.py tests/test_resume_and_parallelism.py
git diff --check
```

Result: both commands exited `0`; all **5 changed Python files** compiled and no whitespace error was reported. Git emitted only its existing LF-to-CRLF working-copy warning for `QCNN/utils/exclusive_queue.py`.

Final passing test executions across the four requested pytest commands: **266 passed, 4 documented environmental skips, 0 failed**. Focused tests overlap the complete-suite totals and are reported separately above rather than as unique test cases.

## Concerns

- Per the explicit no-automatic-stale-lock-deletion requirement, an abandoned operation guard still requires manual/operator recovery; owner release intentionally waits rather than deleting it.
- Four suite tests remain skipped for documented host/data conditions. No required queue, campaign lifecycle, completion, resume, or parallelism regression failed.
- The pre-existing untracked `.claude/settings.local.json` remains untouched and excluded from the fix commit.
