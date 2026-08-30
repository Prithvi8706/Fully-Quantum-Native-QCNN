# Cover letter draft - IEEE journal (venue-neutral)

**Target journal:** [Insert selected journal]

**Article type:** [Insert the venue's article type]
**Title:** FQCNN: A Fully Quantum-Native Convolutional Neural Network with Coherent Unitary Pooling

Dear Editor,

We submit the manuscript "FQCNN: A Fully Quantum-Native Convolutional Neural
Network with Coherent Unitary Pooling" for consideration in [selected journal].

The manuscript makes a bounded engineering and algorithmic contribution. It gives
an explicit fully unitary encoding-convolution-pooling-classification pipeline and
proves that the frozen coherent pooling block induces exactly the corresponding
measure-and-condition channel on the retained register. The practical distinction
is removal of mid-circuit measurement, reset, and classical feed-forward for that
specified construction, not recovery of information or an asserted hardware
speedup.

The evaluation is designed for auditability: four predeclared binary tasks from
MNIST, Fashion-MNIST, and KMNIST; five seeds for the proposed model and primary
baselines; identical split and sample identities; uncertainty and paired tests
with Holm correction; pooling-transfer ablation; finite trainability and
expressibility diagnostics; logical and transpiled resource counts; and a
declared local noise check. The proposed model is below the logistic and two-unit
MLP baselines in mean accuracy on all four tasks and above TTN in mean accuracy on
all four, with no significant primary paired difference after correction. We
report this negative result and the dense MLP's larger parameter count rather than
making a superiority claim. All experiments are classical simulations or local
fake-backend/transpilation checks; no real-QPU result is presented.

The source, locks, evidence, tests, and generated PDF are bound to an exact Git
commit by the accompanying package manifest. A persistent public code/evidence
archive and the author declarations will be completed before portal submission.

An earlier conference draft exists. The submitted journal version is a
substantial extension: it formalises the pooling channel result, corrects the
main-path and locality description, adds the predeclared multi-domain repeated-
seed comparison and pooling-transfer evidence, and reports bounded resource,
noise, and reproducibility checks. The earlier single-task result is retained
only as historical provenance and is not presented as the journal's headline
evidence.

Sincerely,

The authors of FQCNN

## Author confirmation required before upload

- [ ] Every listed author approved the manuscript and this submission.
- [ ] The work is not under review elsewhere and prior dissemination is disclosed.
- [ ] Funding, conflicts of interest, employer approvals, and intellectual-property
      clearances are complete.
- [ ] The corresponding author and contact details are confirmed.
- [ ] The selected venue's template, graphics, ethics, AI-disclosure, licence, and
      APC/waiver checks are confirmed.
