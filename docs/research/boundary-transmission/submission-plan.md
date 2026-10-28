# Publication and artifact plan — checked 2026-10-27

**Recommendation:** release a citable reproduction artifact from the existing
GitHub repository, archive it with Zenodo, and submit the reviewed five-page paper
to **arXiv cs.NE**. For journal peer review, **Artificial Life** is my first-choice
audience. This is a recommendation based on the paper's scope, not an assurance of
acceptance. The failed prediction and established mathematics make novelty the
main editorial risk; a proof assistant does not resolve that risk.

No release, archive deposit, editorial message, or submission has been made.

## Venue comparison

| Route | Fit and recommendation | What needs attention |
|---|---|---|
| arXiv cs.NE | Best preprint category: its official scope includes genetic algorithms, artificial life, and adaptive behavior. | Human authorship, endorsement if required, complete source, artifact reference, and moderation. arXiv is not a journal or peer review. |
| Artificial Life, MIT Press | Best journal audience in my judgment: a methods case about interpreting evolutionary change in a digital ecology. | Ask whether a concise methods/negative-result manuscript fits the current article types. Do not present it as a new Price theorem. |
| Evolutionary Computation, MIT Press | Relevant field, but a less convincing first target for the present single-platform audit. | A demonstrated advance in an evolutionary method or broader comparative study would make the pitch stronger. This is my assessment, not an official rejection criterion. |
| JOSS | Not recommended for this manuscript. | JOSS evaluates substantial research software rather than a paper centered on new results; a one-study helper is a weak fit. |
| GitHub + Zenodo | Best companion artifact route; use the existing `Mirrorad1/active_monkey` repository. | A pinned release and archival DOI make the evidence citable. Hosting does not establish scientific significance or peer review. |

Sources: [arXiv category taxonomy](https://arxiv.org/category_taxonomy),
[arXiv moderation](https://info.arxiv.org/help/moderation/index.html),
[JOSS submission requirements](https://joss.readthedocs.io/en/latest/submitting.html),
and [GitHub's Zenodo instructions](https://docs.github.com/en/repositories/archiving-a-github-repository/referencing-and-citing-content).

For topical context, the Artificial Life editors' [field taxonomy/editorial](https://direct.mit.edu/artl/article-pdf/30/1/1/2354665/artl_e_00435.pdf)
discusses digital evolution and measurement. The official
[Artificial Life submission-guidelines page](https://direct.mit.edu/artl/pages/submission-guidelines)
returned HTTP 403 during this check. Therefore current submission portal, exact
article-type limits, fees, and AI/preprint policies remain to be checked with the
journal before submitting. An editor-authored [2022 template](https://www.overleaf.com/latex/templates/artificial-life-journal-submission-template/zhhhjdgvhryt)
mentions letters of at most 2,000 words, but that is historical guidance for accepted
papers, not a verified current five-page submission rule. The publisher currently
lists Artificial Life and Evolutionary Computation as [hybrid open-access journals](https://mitpress.mit.edu/open-access-at-mit-press/journals/);
no specific fee or free-publication promise is made here.

JOSS also requires an open-source license and evidence of substantial software
development and human intellectual contributions. Its current AI policy requires
disclosure and human accountability. Because the question and design here were
delegated to AI, do not claim that a human made those design choices merely to meet
a venue's declaration. A future software paper would be a separate project, not a
format conversion of this research note. [JOSS policy](https://joss.readthedocs.io/en/latest/submitting.html)

## Concrete GitHub/Zenodo route

The connected GitHub metadata confirms that
[`Mirrorad1/active_monkey`](https://github.com/Mirrorad1/active_monkey) is public and
uses `main` as its default branch. Keep the lab history there; a new repository is
unnecessary for this paper. No root license or citation file was found in this
checkout. The owner must choose appropriate code/data/document licenses and supply
accurate authorship before calling the artifact open source or making license grants.

1. Review and integrate the study's commits on the intended release branch. Include
   the original design and both original runs, the additive audit, and Lean sources.
   Resolve any unrelated working changes separately; do not release an arbitrary
   moving `main` checkout as the study snapshot.
2. Add accurate author/affiliation/contact information, license files, and
   `CITATION.cff`. Link the study README from the repository landing page. Add the
   paper identifier once it exists; do not invent a DOI or arXiv number.
3. Connect the public repository to Zenodo and enable archiving before publishing
   a release. Use a study-specific tag such as `exp283-boundary-transmission-v1`
   at the reviewed commit. The proposed name is not an existing tag.
4. Publish the release with the study README as its entry point. GitHub's documented
   integration archives releases and creates a DOI. Confirm the actual archive,
   files, license, and version DOI, then cite that DOI in the manuscript.

Follow the [official integration steps](https://docs.github.com/en/repositories/archiving-a-github-repository/referencing-and-citing-content).
Avoid storing `.lake`, `.venv`, or compiler downloads in release assets. Preserve
`lean-toolchain` and `lake-manifest.json` so proof dependencies remain pinned.

Artifact entry points:

- `docs/research/boundary-transmission/README.md` and the current `.tex`.
- `docs/research/boundary-transmission/lean/`, excluding `.lake/`.
- `ecology/`, the Exp198 configuration source, Exp283 runner/analyzers, and focused tests.
- `experiments/outputs/exp283*`, including original archives and birth audit files.
- The pre-run design, append-only experiment entry, `pyproject.toml`, and `uv.lock`.

## Concrete arXiv route

The source is self-contained: the scatter plot and bibliography are inline. After
human review, supply author details, replace the pending-review sentence with an
accurate completed-review disclosure, and add the archived artifact DOI. Retain the
negative result and modest novelty claim.

1. The corresponding human author registers/logs in and obtains endorsement if
   requested. Select **cs.NE** as primary; no cross-list is necessary for this scope.
2. Upload the current `trait-growth-selection.tex`, alone or in a source ZIP, select
   it as the top-level file, and compile with arXiv's LaTeX processing. Do not upload
   the old standalone PDF as though it were the revised paper.
3. Inspect every compiled page, references, plot, and metadata; confirm the five-page
   constraint in that environment. Keep the large code/data/Lean artifact in the
   citable archive rather than relying on it to repair gaps in the paper's argument.
4. Enter the exact title, authors, abstract, and artifact link; select an authorized
   distribution license, inspect the preview, then submit. Moderation can decline
   work for insufficient originality or significance.

See [arXiv submission instructions](https://info.arxiv.org/help/submit/index.html).
arXiv requires disclosure of substantial generative-AI use and holds human authors
responsible for all contents; AI tools must not be listed as authors.
[Official AI and moderation policy](https://info.arxiv.org/help/moderation/index.html#policy-for-authors-use-of-generative-ai-language-tools)

## Journal fit inquiry draft — not sent

> We are considering submitting “Trait Growth against the Selection Differential:
> A Lineage Audit of Bounded Mutation,” a concise methods case study in a spatial
> artificial-life system. It applies established Price accounting to 48 runs across
> 16 shared seed blocks, reports a failed prospectively specified prediction, and
> shows why living and newborn endpoints can support different descriptive readings.
> The supplement includes reproducible birth records and Lean proofs of the underlying
> mathematics. We claim neither a new selection theorem nor an identified causal
> fitness effect. Would this scope suit a short article or letter in Artificial Life,
> and which current format and word limit should we follow? The work used OpenAI
> Codex for research design, code, proofs, writing, and adversarial review; we would
> also appreciate confirmation of the applicable disclosure requirements.

Send only after the responsible human author has reviewed the work and wishes to
contact the journal. The sender must add their real identity and avoid representing
automated review as independent human peer review.

## Remaining readiness items

Human author identity and responsibility, substantive domain review, licensing,
release/DOI creation, and current journal-specific rules remain unresolved. The
local mathematics, evidence repairs, and manuscript compile are checked; this does
not mean the draft has been accepted, submitted, or certified publication-worthy.
