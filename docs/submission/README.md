# FQCNN IEEE journal draft and submission preparation

The venue is intentionally undecided. The canonical manuscript is the root-level
`fqcnn.tex`, a complete venue-neutral `IEEEtran` journal draft, and `fqcnn.pdf` is
the PDF produced from it by the documented three-pass build. The tracked
`FQCNN conference version (historical).pdf` is the earlier conference artifact for
editorial disclosure; it is not the current result set.

The reproducible release archive is generated from a clean Git commit by
`scripts/build_q1_submission_package.py`; the generated ZIP is deliberately not
committed because it is a release attachment rather than source code.

Included preparation documents:

- `cover-letter.md` - venue-neutral IEEE editor letter draft;
- `author-declarations.md` - plain-language all-author confirmation checklist;
- `data-code-availability.md` - precise data and software availability statement;
- `conference-extension.md` - disclosure of what the journal version adds;
- `citation-quartile-audit.md` - dated Q1-journal-only bibliography audit and
  its database/category caveat;
- `reproducibility.md` - commands, locks, evidence scope, and external-data notes.

The declaration checklist is not another experiment. It asks the authors to
confirm personal or institutional facts that the repository cannot know: who
approved the paper, contribution roles, funding, conflicts, correspondence,
ethics and rights, prior conference disclosure, AI-tool disclosure, and
APC/licence choices. Complete it only after all authors have reviewed the final
PDF.

The manuscript bibliography currently contains only peer-reviewed journal
articles that pass the documented 2024 SJR Q1 filter. Quartiles are
database-, category-, and year-dependent; the selected venue's current JCR/SJR
record must be rechecked immediately before submission. Dataset provenance is
kept through upstream URLs and checksums, not non-journal dataset preprints.

Before upload, the authors must select a venue, adapt the IEEE draft to that
venue's current template and graphics rules, deposit the code/evidence archive in
a persistent public repository, complete the declarations, and decide any APC or
licence. No journal submission or real-QPU execution is performed by this
repository.
