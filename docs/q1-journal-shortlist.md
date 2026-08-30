# Q1 journal shortlist (candidate venues, checked 2026-08-30)

This is a candidate list, not a final venue decision or a guarantee of acceptance.
Quartile labels are category- and database-dependent and must be checked again
immediately before submission. The manuscript's scientific scope remains
simulation-only; no venue choice authorises a QPU job or a submission.

## Recommendation at a glance

**IEEE Transactions on Quantum Engineering (TQE)** is the strongest default for
the current paper: its engineering scope matches the fully unitary architecture,
pooling theorem, resource accounting, and reproducible simulator benchmark; its
current IEEE title-list entry is Q1; and the existing `IEEEtran` journal draft is
the closest format starting point.

**Quantum Machine Intelligence (QMI)** is the best specialist alternative if the
authors want the paper read primarily as a quantum-machine-learning/QNN paper.
**Machine Learning: Science and Technology (MLST)** is attractive if the
benchmark protocol and reproducibility package are the main editorial story.
**Quantum Science and Technology (QST)** is an aspirational option for a broader
quantum-science claim, but its stated selectivity and broad-impact bar make it the
highest editorial-risk choice for this deliberately bounded, simulation-only
result.

## Comparison matrix

| Candidate | Fit for this manuscript | Q1 and venue evidence | Format, access, and length gate | Main adaptation or editorial risk |
|---|---|---|---|---|
| **TQE** | Fully unitary quantum architecture, circuit theorem, quantum-engineering implications, resource analysis, and a reproducible benchmark. | The official IEEE January 2026 title list reports TQE as full OA, JIF 4.6, and Q1. Recheck the authors' institutional JCR category/year record before upload. | Gold OA; TQE currently lists a US$1,995 APC effective 1 January 2024, with possible IEEE/member/country discounts or waivers; no page limit. The current IEEEtran draft is the nearest format baseline. | Confirm corresponding author, ORCIDs, financial support, prior-version disclosure, rights, and final graphics requirements. Keep the contribution framed as an engineering distinction, not a hardware-performance or quantum-advantage claim. |
| **QMI** | Closest specialist scope: quantum AI, QML/QNNs, quantum data preprocessing, image/signal processing, and quantum software. | Publisher reports a 2025 JIF of 4.6 and a 27-day median first decision. Q1 status is conditional on the current indexing category/database/year and must be checked in institutional JCR/SJR records. | Hybrid; subscription publication has no APC, while the current Springer page lists OA at £2,590 / US$3,490 / €2,890. Research articles have no stated fixed length. Convert to Springer guidance: 150–250-word abstract, 4–6 keywords, author-year references, declarations, and data-availability statement. | Springer conversion and reference-style change. The paper must make clear why the coherent construction is useful even though the measured channel is reproduced exactly. |
| **MLST** | Strong fit for quantum computing, neural architectures, codes/datasets, and benchmark studies; the open evidence package is an editorial asset. | Q1 classifications for MLST vary by database and category; treat them as conditional and verify the current target record immediately before submission rather than as a package fact. | Fully OA; current IOP guidance lists a US$3,125 APC and no submission charge, subject to agreements/discounts. Research papers are normally no more than 8,500 words; IOP uses a PDF-first submission workflow and optional templates. | Count words after conversion and retain enough technical detail for the benchmark/reproducibility bar. Keep all negative and inconclusive findings visible rather than optimizing the narrative for a benchmark win. |
| **QST** | Covers quantum computation, QML, software, algorithms, and engineering, but is aimed at work with significant and lasting broad impact. | Q1 appears in relevant current SJR/JCR records, but the exact category/year must be verified at selection. This is not treated as an unconditional release fact. | Hybrid; subscription publication is free, while the current IOP page lists OA at £2,930 / US$4,090 / €3,335. Generic IOP guidance uses a concise title, an abstract up to 300 words, embedded figures, and data/supplement options. | Highest selectivity risk. The current bounded simulator evidence and absence of a hardware result may be a weaker match for the journal's stated lasting-impact bar; do not add unsupported claims to compensate. |

Only TQE's Q1 label is tied here to a current official IEEE title-list source. The
other candidates remain suitable conditional Q1 options whose classification must
be verified in the target indexing system and category at the time of selection.
Costs, discounts, and review-time metrics are also time-sensitive and are not
promises about editorial outcome.

## Primary candidate: IEEE Transactions on Quantum Engineering (TQE)

**Status:** candidate only; the current manuscript is venue-neutral IEEEtran.

- **Q1 evidence:** IEEE's January 2026 title list reports TQE (ISSN 2689-1808) as
  full open access, JIF 4.6, and Q1. This is the Q1 basis recorded here; the
  corresponding category and year should be rechecked in the authors' institutional
  Journal Citation Reports account before upload.
- **Scope:** TQE accepts regular, review, and tutorial articles on engineering
  applications of quantum phenomena, including quantum computation, quantum
  software, algorithms, hardware, and devices. A fully unitary QCNN architecture,
  circuit theorem, resource analysis, and reproducible benchmark fit this scope.
- **Publication model:** gold open access and no page limit. The TQE submission page
  currently displays a US$1,995 APC effective 1 January 2024; IEEE's current
  billing page is authoritative at submission because the journal page and general
  IEEE fee schedules can change. IEEE member and country-based discounts/waivers
  may apply.
- **Format gate:** use `\\documentclass[journal]{IEEEtran}` and the TQE author
  template, submit a public-presentable PDF plus source and individual graphics,
  and remove TODOs, private notes, and local paths. The repository build is a
  journal-style IEEEtran review draft; the author portal's final-file checks remain
  external gates.
- **Submission gate:** obtain all-author approval, conflict/funding declarations,
  rights clearance, and a final APC decision before using the IEEE portal. No
  submission has been performed by this repository workflow.

Sources: [TQE submission process and scope](https://tqe.ieee.org/submission-process/),
[IEEE January 2026 title list](https://open.ieee.org/wp-content/uploads/IEEE-Title-List-January-2026.pdf),
and the [TQE author template](https://journals.ieeeauthorcenter.ieee.org/wp-content/uploads/sites/7/TQE_Template_v4.pdf).

## Fallback 1: Quantum Machine Intelligence (Springer Nature)

This is the closest specialist QML fallback. Springer lists the journal as hybrid,
with a scope spanning theoretical and experimental quantum computing/AI, and reports
a 2025 JIF of 4.6 and a 27-day median to first decision. The exact current quartile
and APC should be verified through institutional JCR and Springer’s fee checker;
neither is treated as a release fact here. Reformatting would require the Springer
submission class and author-guideline checks, so it is not the current build target.

Source: [Quantum Machine Intelligence journal page](https://link.springer.com/journal/42484).

## Fallback 2: Machine Learning: Science and Technology (IOP)

MLST explicitly includes quantum computing, new neural-network architectures, codes,
and benchmark studies. Research papers are normally no more than 8,500 words. It is
fully open access; the current publishing-support page lists a US$3,125 APC and no
submission charge, subject to agreements and discounts. Its current quartile must be
checked in the target indexing system at submission (the Q1 label is not assumed in
this package). The paper would need an IOP manuscript conversion and a tighter
benchmark-oriented framing.

Source: [MLST scope, article types, and charges](https://publishingsupport.iopscience.iop.org/journals/machine-learning-science-and-technology/about-machine-learning-science-and-technology/).

## Fallback 3: Quantum Science and Technology (IOP)

QST covers quantum computation, quantum machine learning, software, algorithms, and
engineering, but describes itself as highly selective and expects a significant,
lasting advance. It is hybrid; subscription publication is free and the current OA
APC is listed as US$4,090. This is an aspirational fallback, not an implied
acceptance prediction. Its current quartile must be verified before use.

Source: [QST scope, selectivity, and charges](https://publishingsupport.iopscience.org/journals/quantum-science-technology/about-quantum-science-technology/).

## Decision and adaptation order

1. Prepare and internally review the venue-neutral IEEE journal package.
2. Select a venue, then recheck its JCR category, APC, author template, ethics, and
   all-author consent at the intended submission date.
3. If TQE is selected, adapt the package to its current requirements; otherwise
   adapt the same evidence package to QMI,
   then MLST, then QST. Do not alter the reported results merely to fit a venue.
4. Keep the claims bounded: no quantum advantage, hardware validation, universal
   trainability, image-space locality, or broad-domain generalisation claim is
   supported by this package.

Useful official cross-checks are the [IEEE article structure guidance](https://journals.ieeeauthorcenter.ieee.org/create-your-ieee-journal-article/create-the-text-of-your-article/structure-your-article/),
[IEEE graphics guidance](https://journals.ieeeauthorcenter.ieee.org/create-your-ieee-journal-article/create-graphics-for-your-article/file-formatting/),
[QMI aims and scope](https://link.springer.com/journal/42484/aims-and-scope),
[QMI submission guidelines](https://link.springer.com/journal/42484/submission-guidelines),
[MLST author guidance](https://publishingsupport.iopscience.iop.org/journals/machine-learning-science-and-technology/),
and [QST author guidance](https://publishingsupport.iopscience.iop.org/journals/quantum-science-and-technology/).

No journal submission, upload, authentication, or real-QPU execution has been
performed.
