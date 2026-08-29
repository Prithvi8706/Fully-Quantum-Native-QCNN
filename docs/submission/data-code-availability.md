# Data and code availability statement

The FQCNN implementation, tests, manuscript source, environment locks, evidence
JSON, per-cell comparison manifests, and reproduction instructions are maintained
in the version-controlled repository at the release commit recorded in the package
manifest. The package contains SHA-256 records for every included file.

The experiments use the upstream MNIST, Fashion-MNIST, and KMNIST IDX sources. The
downloaded datasets are not redistributed in this repository. Their source URLs,
labels, versions, local paths, and expected/observed checksums are recorded in
`Results/evidence/q1_dataset_provenance.json`; the local MNIST snapshot is
checksum-pinned and is intentionally not fetched by the registry. Users should
obtain each dataset from its cited upstream source and verify the recorded hashes
before reproducing a run.

The aggregate evidence is simulation-only. Ignored run-weight and prediction
archives are regenerable and are not silently represented as a public dataset
deposit. Before journal upload, the authors must deposit the code/evidence archive
in a persistent public repository and replace this paragraph's repository pointer
with its DOI or immutable URL. A journal submission has not been made by this
workflow.
