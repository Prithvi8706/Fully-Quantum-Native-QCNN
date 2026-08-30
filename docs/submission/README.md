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
- `reproducibility.md` - commands, locks, evidence scope, and external-data notes.

The declaration checklist is not another experiment. It asks the authors to
confirm personal or institutional facts that the repository cannot know: who
approved the paper, contribution roles, funding, conflicts, correspondence,
ethics and rights, prior conference disclosure, AI-tool disclosure, and
APC/licence choices. Complete it only after all authors have reviewed the final
PDF.

Before upload, the authors must select a venue, adapt the IEEE draft to that
venue's current template and graphics rules, deposit the code/evidence archive in
a persistent public repository, complete the declarations, and decide any APC or
licence. No journal submission or real-QPU execution is performed by this
repository.
