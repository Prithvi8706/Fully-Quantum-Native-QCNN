# Citation-quartile audit

**Audit date:** 2026-08-30  
**Manuscript:** `paper/fqcnn.tex`  
**Policy:** every bibliography item cited by the manuscript is a peer-reviewed
journal article. The retained venue is reported as **Q1 in the 2024 SCImago
Journal & Country Rank (SJR/Scopus) data in a category relevant to quantum
computing, physics, engineering, or control**. No arXiv-only item, book, or
conference proceeding is retained in the bibliography.

Quartiles are properties of a journal, database, subject category, and year;
they are not properties of an individual article. JCR and SJR can therefore
disagree. The policy above is the reproducible filter used for this draft, not
a guarantee of a particular institution's JCR classification or of acceptance
by a Q1 venue. Before submission, recheck the selected venue and any
institution-specific JCR/SJR rule using the current year's official record.

## Retained references

| Keys | Venue | 2024 Q1 evidence | Role in the manuscript |
|---|---|---|---|
| `ref14` | PRX Quantum | SJR Q1 in quantum/physics and related categories | expressibility and gradient-scale context |
| `ref15`, `ref16`, `ref23`, `ref46` | npj Quantum Information | JCR/SJR Q1 in quantum-science and related physics categories | stochastic optimisation, noise mitigation, autoencoders, and hierarchical classifiers |
| `ref19` | Physical Review Letters | SJR/JCR Q1 in physics | expressivity context |
| `ref21` | Physical Review Research | SJR/JCR Q1 in physics | stochastic optimisation context |
| `ref22` | Reviews of Modern Physics | SJR Q1 in physics | error-mitigation review |
| `ref25` | Quantum Science and Technology | SJR/JCR Q1 in quantum science and technology | quantum autoencoder context |
| `ref26` | Automatica | SJR Q1 in control/automation | quantum-autoencoder compression context |
| `ref37` | Nature Physics | SJR/JCR Q1 in physics | canonical QCNN and measurement-based pooling context |
| `ref38`, `ref47` | Nature Communications | SJR/JCR Q1 in multidisciplinary physics | barren-plateau context |
| `ref40` | Nature Reviews Physics | SJR Q1 in physics | variational-circuit review |
| `ref49` | Proceedings of the IEEE | SJR/JCR Q1 in computer science/engineering | MNIST provenance; this is a peer-reviewed IEEE journal, not a conference proceeding |

The quartile records used for this audit include the [2024 SCImago physics
rankings](https://www.scimagojr.com/diamond/journalrank.php?category=3101&country=Northern+America&type=j),
[PRX Quantum's 2024 SJR listing](https://kniznica.umb.sk/app/cmsSiteBoxAttachment.php?ID=6377&cmsDataID=0),
[npj Quantum Information's 2024 metrics](https://jrank.net/journals/npj-quantum-inform/metrics),
[Quantum Science and Technology's 2024 metrics](https://jrank.net/journals/quantum-sci-technol/metrics),
and the publisher/indexing records for the retained venues. These links are
audit evidence, not manuscript citations.

## Explicit exclusions

The following entries were removed from the manuscript bibliography and their
claims were either narrowed or supported by the retained journal literature:

- arXiv-only preprints: `ref2`, `ref3`, `ref4`, `ref6`, `ref7`, `ref8`, `ref10`,
  `ref11`, `ref13`, `ref29`, `ref36`, `ref52`, and `ref53`;
- non-journal book: `ref51`;
- venues with a relevant-category JCR/SJR classification that is conditional
  or mixed for this policy: `ref17`, `ref18`, `ref20`, `ref41`, `ref42`,
  `ref45`, and `ref50`.

Fashion-MNIST and KMNIST remain reproducible dataset inputs. Their upstream
URLs, labels, snapshot identities, and checksums are recorded in
`Results/evidence/q1_dataset_provenance.json`; canonical dataset preprints are
not cited because this manuscript follows the strict Q1-journal-only policy.

## Mechanical checks

The final source contains 15 cited bibliography keys and 15 matching entries.
The bibliography contains no `arXiv`, book, or excluded conditional venue
entry. Re-run the citation-key check after any manuscript edit, then repeat the
quartile check immediately before selecting and submitting to a venue.
