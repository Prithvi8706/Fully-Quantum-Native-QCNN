# Reproduction and release instructions

The supported training environment is Python 3.9.13 with the exact packages in
`requirements-lock.txt`. Qiskit/fake-backend checks use the separate Python 3.11
environment described by `requirements-qiskit-lock.txt`; do not mix those locks.

From a clean checkout of the release commit:

```text
python -m pip install -r requirements-lock.txt
python -m pytest tests -q
pdflatex -interaction=nonstopmode -halt-on-error fqcnn.tex
pdflatex -interaction=nonstopmode -halt-on-error fqcnn.tex
pdflatex -interaction=nonstopmode -halt-on-error fqcnn.tex
python -m ruff check .
python -m compileall -q QCNN baselines experiments scripts tests
```

The committed aggregate JSON is validated against its per-cell manifests and source
hashes. Re-running the full training campaign is not required to inspect the
submitted claims; if it is repeated, use the frozen task matrix, source-stable IDs,
666-example pools, 400/100/166 split, and seeds `0..4` recorded in the manifests.
Downloaded data and ignored run archives are external, regenerable inputs and must
not be substituted with a different split.

After the final PDF and source commit are ready, run:

```text
python scripts/build_q1_submission_package.py --output-dir submission_dist
```

The command refuses tracked edits, stale/non-PDF manuscript output, unsafe archive
paths, and common credential signatures. It writes a ZIP containing the source,
figures, evidence, locks, tests, and preparation documents, then reopens the ZIP
and verifies every recorded SHA-256 and byte count. The archive is an attachment or
release artifact, not a Git-tracked build output.
