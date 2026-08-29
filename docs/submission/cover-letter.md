# Cover letter draft — IEEE Transactions on Quantum Engineering

**Article type:** Regular Article  
**Title:** FQCNN: A Fully Quantum-Native Convolutional Neural Network with Coherent Unitary Pooling

Dear Editor,

We submit the manuscript “FQCNN: A Fully Quantum-Native Convolutional Neural
Network with Coherent Unitary Pooling” for consideration as a Regular Article in
*IEEE Transactions on Quantum Engineering*.

The manuscript makes a bounded engineering and algorithmic contribution. It gives
an explicit fully unitary encoding–convolution–pooling–classification pipeline and
proves that the frozen coherent pooling block induces exactly the corresponding
measure-and-condition channel on the retained register. The practical distinction
is removal of mid-circuit measurement, reset, and classical feed-forward for that
specified construction—not recovery of information or an asserted hardware speedup.

The evaluation is designed for auditability: four predeclared binary tasks from
MNIST, Fashion-MNIST, and KMNIST; five seeds for the proposed model and primary
baselines; identical split/sample identities; uncertainty and paired tests with
Holm correction; pooling-transfer ablation; finite trainability and expressibility
diagnostics; logical/transpiled resource counts; and a declared local noise check.
The proposed model is below the logistic and two-unit MLP baselines in mean
accuracy on all four tasks and above TTN in mean accuracy on all four, with no
significant primary paired difference after correction. We report this negative
result and the dense MLP's larger parameter count rather than making a superiority
claim. All experiments are classical simulations or local fake-backend/transpilation
checks; no real-QPU result is presented.

The source, locks, evidence, tests, and generated PDF are bound to an exact Git
commit by the accompanying package manifest. The authors will provide a persistent
public code/evidence archive and complete the declarations below before portal
submission.

Sincerely,  
The authors of FQCNN

## Author confirmation required before upload

- [ ] Every listed author approved the manuscript and this submission.
- [ ] The work is not under review elsewhere and prior dissemination is disclosed.
- [ ] Funding, conflicts of interest, employer approvals, and intellectual-property
      clearances are complete.
- [ ] The corresponding author and contact details are confirmed.
- [ ] APC funding/waiver status and the final TQE template/graphics checks are
      confirmed.
