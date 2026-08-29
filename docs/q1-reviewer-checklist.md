# Q1 reviewer-risk checklist

This finite checklist is the release gate for the current FQCNN manuscript. It
records the evidence a reviewer can inspect, the remaining risk, and the one
bounded correction pass. `Pass with caveat` means the limitation is stated in the
paper; it is not permission to broaden the claim.

| # | Reviewer question | Evidence / pointer | Severity | Status |
|---:|---|---|---|---|
| 1 | Is the canonical architecture unchanged and demonstrably unitary? | `tests/test_freeze_architecture.py`; `tests/fixtures/headline_signature.json`; `fqcnn.tex` Secs. III–IV | High | Pass |
| 2 | Are three dataset families and all four tasks represented without test-driven pair selection? | `Results/evidence/q1_dataset_provenance.json`; `Results/evidence/q1_comparison.json` protocol | High | Pass |
| 3 | Are split identities, ordered IDs, seed counts, uncertainty, and one final test evaluation visible? | `Results/q1_comparison/manifests/`; per-cell JSON; aggregate `protocol` and `artifact_hashes` | High | Pass |
| 4 | Are classical and hierarchical quantum baselines evaluated on identical data? | Aggregate split/sample identity validation; `baselines/classical_cnn.py`; `baselines/quantum_baselines.py` | High | Pass with caveat: the two-unit dense MLP has 1,573 parameters, so it is not called parameter-matched; the conservative capacity mismatch is disclosed. |
| 5 | Are paired tests, assumptions, and multiplicity control reported? | `Results/evidence/q1_comparison.json` `proposed_vs`; Holm–Bonferroni family of 12 tests | High | Pass |
| 6 | Does the pooling ablation separate equivalence, architecture, and accuracy? | `Results/evidence/e1_pooling_equivalence.json`, `e2_dephasing.json`, `q1_pooling_transfer.json`, `q1_pooling_controls.json` | High | Pass |
| 7 | Are allocated, tape-reaching, and effective parameters distinguished? | `tests/fixtures/effective_params.json`; `fqcnn.tex` classifier section | Medium | Pass: 269 / 78 / 74 |
| 8 | Is the full DLA result reported without a trainability certificate? | `Results/evidence/t6_dynamical_lie_algebra.json`; `docs/simulability_statement.md` | High | Pass |
| 9 | Are state-preparation, model-body, readout, and total resources separated? | `Results/evidence/q1_resources.json`; `fqcnn.tex` practicality section | Medium | Pass |
| 10 | Does zero-noise output agree with the canonical path, without calling local noise device validation? | `Results/evidence/q1_noise_validation.json`; `q1_fake_backend_rehearsal.json` | High | Pass with simulation/local-device scope |
| 11 | Are simulator, fake backend, local transpilation, and real hardware distinct? | `README.md`; `fqcnn.tex` limitations; `Results/evidence/q1_fake_backend_rehearsal.json` | High | Pass: no real-QPU job is claimed or submitted |
| 12 | Are advantage, locality, scalability, and robustness claims no stronger than evidence? | `fqcnn.tex` introduction, results, limitations; reconciliation table | High | Pass |
| 13 | Does every numerical manuscript claim trace to committed evidence? | `Results/evidence/q1_claim_ledger.json`; source/artifact SHA-256 records | High | Pass after final ledger refresh |
| 14 | Can a clean checkout validate evidence, run tests, and build the manuscript? | `Results/evidence/q1_reproduction.json`; `scripts/build_q1_submission_package.py` | High | Pass: 631 tests (627 passed, 4 documented skips), clean training/Qiskit checks, three-pass LaTeX build |
| 15 | Are limitations, negative results, and conditional hardware status explicit? | `fqcnn.tex` Sec. V/VI; `docs/submission/data-code-availability.md` | Medium | Pass |

## One correction pass

The final pass explicitly resolved five concrete risks before release:

1. the SWAP strict-containment witness now includes the global phase needed for
   membership in `SU(4)`;
2. the coherence functional uses the trace norm consistently (`1`, with trace
   distance `1/2`);
3. the terminal `R_Z` is documented as structurally inert, not as a trainable
   bias;
4. parameter-shift and finite-shot BCE-gradient wording is scoped correctly; and
5. the dense baseline is called a two-unit MLP with its 1,573-parameter capacity,
   never a falsely parameter-matched comparator.

## External pre-upload gates

These are intentionally outside the repository's completion claim: all-author
approval and contribution statements, conflict/funding details, a public code/data
archive with a DOI, final IEEE TQE template/graphics checks, and the APC decision.
They must be completed by the authors before using a journal portal. They do not
justify changing the scientific evidence or claiming that a submission has already
occurred.
