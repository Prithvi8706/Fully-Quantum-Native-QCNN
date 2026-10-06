# FQCNN venue-neutral journal draft and submission preparation

The venue is intentionally undecided. The canonical manuscript is the root-level
`paper/fqcnn.tex`, a complete venue-neutral two-column `IEEEtran` manuscript, and
`paper/fqcnn.pdf` is the PDF produced from it by the documented three-pass build.

The reproducible release archive is generated from a clean Git commit by
`scripts/build_q1_submission_package.py`; the generated ZIP is deliberately not
committed because it is a release attachment rather than source code.

The current candidate comparison is [`docs/q1-journal-shortlist.md`](../q1-journal-shortlist.md).
No journal has been selected, so no journal template conversion has been made.

Included preparation documents:

- `cover-letter.md` - venue-neutral editor letter draft;
- `author-declarations.md` - plain-language all-author confirmation checklist;
- `data-code-availability.md` - precise data and software availability statement;
- `citation-quartile-audit.md` - dated Q1-journal-only bibliography audit and
  its database/category caveat;
- `reproducibility.md` - commands, locks, evidence scope, and external-data notes.

The declaration checklist is not another experiment. It asks the authors to
confirm personal or institutional facts that the repository cannot know: who
approved the paper, contribution roles, funding, conflicts, correspondence,
ethics and rights, AI-tool disclosure, and
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

## Venue-neutral readiness audit (checked 2026-08-31)

The audit below records what is ready now and what deliberately remains a
post-selection action. It does not change the validated numerical results or
rerun the completed experiments.

| Area | Current status | Evidence or remaining action |
|---|---|---|
| Title | Ready as a neutral baseline | `paper/fqcnn.tex` uses a specific title without a journal name or an unsupported novelty/advantage claim. Recheck only the selected venue's title-length/style preference. |
| Abstract | Ready as a neutral baseline | One paragraph, 173 words, self-contained, with defined `QCNN`/`FQCNN` terms and no citations, equations, figure/table references, or hardware-performance claim. Recheck the selected venue's word and acronym rules. |
| Keywords | Ready as a neutral baseline | Four terms are present and alphabetized for the current IEEE draft. Convert count, separators, and index-term style after selection. |
| References | Ready under the project policy | 15 cited entries match the bibliography; every retained entry has a DOI and passes the documented 2024 SJR Q1 journal-only filter. Convert only the citation style required by the selected venue. |
| Figures and tables | Ready for venue-neutral review | Four manuscript figures are referenced and present, with 21 table captions and 4 figure captions (25 total). `fig1.png`, `fig3.png`, and the finalized `fig4.png` report about 600 dpi; `q1_comparison.png` reports about 300 dpi. Recheck the selected venue's line-art, color, naming, and embedded-font rules after selection. |
| Page/word limit | Venue-dependent | The IEEEtran PDF is 17 US-letter pages. TQE lists no page limit; QMI lists research articles at arbitrary length; MLST normally limits research papers to 8,500 words; QST's article-type gate must be checked after selection. The 17-page count is a venue-neutral working length, not a claim that every journal permits it. |
| Author metadata | Confirmation required | The manuscript lists four authors (Aasa Singh Bhui, Prithvi Raghu, J Jayashree*, J Vijayashree) with VIT affiliations and institutional email addresses. Confirm author order, the corresponding author marked `*`, all-author approval, ORCIDs, affiliations, and the final contact address; no ORCID has been invented. |
| Declarations | Confirmation required | [`author-declarations.md`](author-declarations.md) contains the checklist, but funding, conflicts, contributions, ethics/rights, AI disclosure, correspondence, and APC/licence facts remain blank until the authors confirm them. |
| Data and code | Statement ready; archive action open | [`data-code-availability.md`](data-code-availability.md) states the simulation-only scope and upstream-data provenance. Deposit the release archive in a persistent public repository after selection and replace the placeholder with its DOI/immutable URL. |
| Cover letter | Draft ready; target fields open | [`cover-letter.md`](cover-letter.md) contains the bounded contribution, exact result framing, and negative findings. Insert the selected journal, article type, editor-facing fit, and final archive/declaration facts only after selection. |

The figure-resolution action above was a submission-quality operation, not a
request to repeat an experiment: it concerns only the bitmap encoding of an
already validated diagram. The current manuscript source remains venue-neutral;
template and venue-specific edits remain deferred until selection.

## Post-selection checklist

After the authors select a journal, complete these source-level actions in order:

1. Recheck the journal's current quartile category, article type, template,
   abstract/keyword rules, reference style, page or word limit, graphics rules,
   APC/licence, and declarations against its official author pages.
2. Adapt `paper/fqcnn.tex` and the bibliography to that template without changing
   validated data, statistical results, or claim boundaries.
3. Regenerate or validate figures at the selected journal's resolution, naming,
   color, and embedded-font requirements; inspect the complete PDF and page/word
   count.
4. Complete all-author metadata, CRediT/contribution text, funding and competing
   interests, ethics/rights, AI/tool disclosure,
   data/code DOI, and licence/APC choices.
5. Run the final build, evidence, test, secret, and package gates from a clean
   commit; obtain all-author approval of the exact upload files.
6. Stop for explicit user approval before any external upload, authentication,
   journal submission, or real-QPU execution.
