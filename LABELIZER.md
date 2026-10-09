# LeREaD Labelizer — Annotation Tool

This document describes the **LeREaD Labelizer** tool used for the manual annotation process. It is written as a companion to the article: the detailed criterion-by-criterion evaluation was removed from the article for length, and is reported here in full.

No external links are included, in keeping with the anonymized review version.

## 1. Motivation

General-purpose annotation platforms (e.g. BRAT, doccano, INCEpTION, YEDDA) cover a wide range of NLP corpus-creation workflows, from lightweight desktop tools to web-based collaborative systems with model-assisted annotation. Selecting an appropriate tool remains non-trivial and often requires substantial evaluation against domain-specific format, workflow, and compatibility constraints.

Our annotation scheme introduces requirements only partially supported by existing tools:

- **Hierarchical structured labels** decomposed into semantically distinct components (titles, fragments, references) with associated attributes (e.g. title type, fragment identifiers).
- **Explicit co-reference modelling** linking all surface mentions of the same authority via a shared `docid` identifier.
- **Dedicated machine-vs-human separation** via `<auto_label>` tags for machine-generated annotations versus `<manual_label>` tags for human annotations.
- **Direct work on full-length legal HTML** with layout preserved, rather than plain-text import / export round-trips.
- **Accessibility for domain experts** with limited technical background: fully client-side, no installation, improving findability and reusability in the sense of the FAIR principles.

Rather than replacing existing platforms, the LeREaD Labelizer addresses these format- and task-specific requirements.

## 2. Tool overview

The LeREaD Labelizer is a client-side HTML annotation tool:

- **No installation required.** A document is opened directly in the tool and annotated in place.
- **Fully local usage.** Documents never leave the annotator's machine, ensuring data privacy.
- **Native HTML input.** XML and plain text are also supported. Annotated documents are saved back as HTML (XML export supported).
- **UTF-8 multilingual support.** Required for the bilingual (English / French) corpus.
- **Full-text documents of arbitrary length.** No truncation or pre-segmentation is imposed on annotators.
- **Partial annotation and later continuation.** Work can be saved at any time (`Save As` under the same name and location) and resumed. An optional automatic version history can store a snapshot of the document at regular intervals.
- **Text highlighting.** All annotations are visualised in situ with per-label colours.
- **Free and open-source**, with web-based documentation, usable with full functionality at no cost.

## 3. Input format

Before a document can be loaded, an HTML comment containing the agreed label tree and project metadata is injected at the top of the file (before `<head>`):

```html
<html><!-- HTMLLabelizer
{
  "labeltree": { ... },
  "meta": { ... }
}
--><head> ...
```

- `labeltree` is a standard JSON object defining top-level labels, sublabels, colours, and attributes (see Section 4).
- `meta` records project title, label-tree version, annotation guidelines version, annotator, reviewer, dates, and elapsed time.

This pre-processing step converts source files to UTF-8 and makes the annotation schema configurable per project without changing the tool itself.

## 4. Label scheme

Four top-level authority types are annotated. Each carries a `docid` (co-reference cluster identifier) and a `uri` (resolution target), plus nested structural sublabels:

| Top-level label | Meaning | Sublabels | Attributes (top-level) |
|---|---|---|---|
| `legislation` | Mention of legislation | `title`, `citation`, `fragment` | `docid`, `uri` |
| `decision` | Mention of a judicial or administrative decision | `title`, `citation`, `fragment` | `docid`, `uri` |
| `secondary sources` | Legal scholarship and other secondary sources (reports, dictionaries, talks, working papers) | `authors`, `title`, `source`, `fragment` | `docid`, `uri` |
| `unable to classify` | Exceptional borderline case (legal source vs. pure factual evidence) flagged for later review | — | `docid`, `uri` |

Sublabel details:

| Sublabel | Use | Attributes |
|---|---|---|
| `title` | Official full title or alias (short title, acronym, contextual reference such as "the Act") | `titletype`: dropdown `official` / `alias` |
| `citation` | Bibliographic / publication reference for legislation and decisions. Each parallel citation gets its own tag. | — |
| `source` | Bibliographic / publication information for secondary sources (journal, publisher, volume, year). Partial information is included. | — |
| `authors` | Author(s) of a secondary-source contribution. All authors including "et al." in one tag; volume editors go in `source` unless they are also contribution authors. | — |
| `fragment` | Precise cited portion: section / subsection / paragraph / page, etc. | `fragmentid` (string, normalised, e.g. `sec 3(2)(a)`, `para 25`, `p 353`, `p 353 - p 362`), `non_standard` (checkbox for formats outside the normalised nomenclature) |

General tagging rules:

- Every mention is tagged, whether it is a complete reference (all bibliographic elements) or a reduced / partial form (title only, alias, isolated fragment, Latin back-references such as `supra`, `ibid`, `id`).
- General tags exclude trailing punctuation and a leading isolated "at" introducing a fragment-only mention. Specific sublabels likewise exclude internal punctuation of the enclosing general tag.
- Specific sublabels within two stop-words of each other are kept in the same general tag; beyond that they are split into distinct general tags.

## 5. Annotation workflow

1. **Load.** Open the prepared HTML document with the `load HTML file` function.
2. **General tags.** Select the full span of each authority mention and assign `legislation`, `decision`, `secondary sources`, or `unable to classify`.
3. **Specific tags.** Inside each general tag, tag the elements present: `authors`, `title` (+ `titletype`), `citation` / `source`, `fragment` (+ `fragmentid`, `non_standard` where applicable).
4. **Co-reference (`docid`).** Manually assign a `docid` per authority and systematically reuse the same `docid` for every mention of that authority in the document, even when surface forms vary. Assigned identifiers appear in the `Co-reference` pane; clicking a `docid` highlights all its mentions for verification. `docid` conventions distinguish shortened titles, historical versions of the same act, same-title acts from different jurisdictions, same-party decisions, and same-title publications.
5. **Resolution (`uri`).** In the `Co-reference` pane, paste the short identifier of the cited authority for the corresponding `docid`, or `None` when the authority is not available in the reference collection. For contributions published only as part of a larger volume, the contribution-level identifier is used.
6. **Save.** Save with a neutral-reference filename including annotator initials, in the annotator's subfolder. Re-save regularly with `Save As`; optional auto-save keeps a timed version history.

Practical aids:

- Adjust a tag boundary: hold `Ctrl`, click the tag, then click the new boundary.
- Delete a tag: right-click it.
- Annotate several equivalent mentions jointly: hold `Ctrl` to multi-select equivalent spans, then apply the same tags once; an `advanced labelization` helper is also available (its selection must be reviewed).
- Quoted extracts: tag additional authorities cited inside a block quote, but do not tag the structural elements of the quote itself.
- Special cases (acts embedded in other acts, constitutional texts, foreign / international authorities, `Indexed as` headings, URLs inside doctrinal references) are each tagged as distinct authorities.

## 6. Output format

Annotations are stored inline as nested tags preserving the original HTML layout, for example:

```html
<manual_label labelname="decision" docid="Vavilov SCC 2019" uri="..." verified="false">
  <manual_label labelname="title" parent="decision" titletype="official" verified="false">Canada (Minister of Citizenship and Immigration) v. Vavilov</manual_label>,
  <manual_label labelname="citation" parent="decision" verified="false">2019 SCC 65</manual_label>
</manual_label>
```

- Human annotations use `<manual_label>`; machine-generated pre-annotations use `<auto_label>` with the same `labelname` mechanism, keeping the two provenances explicitly separable.
- Styling (background colour per label) is stored inline for visualisation; the semantic content is carried by `labelname`, `parent`, `docid`, `uri`, `titletype`, `fragmentid`, `non_standard`, `id`, and `verified` attributes.
- Because schema, colours, and attributes travel with the file header, outputs remain self-describing.

The same inline representation supports pre-annotation import (external suggestions can be loaded and corrected), inter-annotator agreement comparison, and downstream extraction / co-reference / resolution processing.

## 7. Evaluation

The tool was assessed with the evaluation framework for HTML annotation tools that scores publication (P), technical (T), data-format (D), and functional (F) criteria, each criterion contributing 0.0, 0.5, or 1.0.

### 7.1 Detailed scoring

| Criterion | Score | Justification |
|---|---:|---|
| P1 | 1.0 | Last version released in 2026 |
| P2 | — | Not applicable |
| P3 | — | Not applicable |
| T1 | 1.0 | Latest version released in 2026 |
| T2 | 1.0 | Source code publicly available |
| T3 | 1.0 | Online availability for direct use |
| T4 | 1.0 | No installation required |
| T5 | 1.0 | Web-based documentation available |
| T6 | 1.0 | Open-source license allowing unrestricted use |
| T7 | 1.0 | Freely available with full functionality |
| D1 | 1.0 | Configurable schema using standard JSON format |
| D2 | 1.0 | Native HTML input (XML and plain text supported) |
| D3 | 1.0 | Annotations downloadable as HTML or XML |
| F1 | 1.0 | Multi-label annotation via hierarchical label tree |
| F2 | 0.0 | No support for document-level annotation |
| F3 | 0.5 | Limited relationship support via clustering identifiers |
| F4 | 0.0 | No ontology or terminology integration |
| F5 | 0.5 | Support for importing external pre-annotations |
| F6 | 0.0 | No integration with PubMed / Medline |
| F7 | 1.0 | Support for full-text documents of arbitrary length |
| F8 | 1.0 | Partial annotation and later continuation supported |
| F9 | 1.0 | Text highlighting fully supported |
| F10 | 0.0 | No multi-user or team management |
| F11 | 1.0 | Inter-annotator agreement supported |
| F12 | 1.0 | Fully local usage ensuring data privacy |
| F13 | 1.0 | Support for UTF-8 encoded multilingual documents |
| **Total** | **19 / 24** | **Normalized score: 0.79** |

Notes on the low / partial scores:

- **F2 (0.0):** the scheme is span-based by design; there is no document-level classification layer.
- **F3 (0.5):** relations are modelled only as co-reference clusters through the shared `docid`, not as typed arbitrary relations.
- **F4 (0.0) / F6 (0.0):** no ontology service or biomedical-literature integration, consistent with a legal-citation task.
- **F5 (0.5):** pre-annotations can be imported and corrected (used for the LLM-assisted workflow), but there is no built-in active-learning loop.
- **F10 (0.0):** single-user local tool; collaboration is handled by file exchange and a reviewer pass rather than user / team management.

### 7.2 Comparison with surveyed tools

Scores below reuse the published survey results for four representative tools under the same framework. `P`, `T`, `D`, `F` are category subtotals.

| Tool | P | T | D | F | Total | Score |
|---|---|---|---|---|---|---|
| BioQRator | 1.5 | 4.5 | 3.0 | 6.0 | 15.0 | 0.58 |
| brat | 3.0 | 5.0 | 2.0 | 9.5 | 19.5 | 0.75 |
| FLAT | 0.0 | 7.0 | 3.0 | 8.5 | 18.5 | 0.71 |
| WebAnno | 3.0 | 5.0 | 3.0 | 10.0 | 21.0 | 0.81 |
| **LeREaD Labelizer (ours)** | **1.0** | **7.0** | **3.0** | **8.0** | **19.0** | **0.79** |
| Possible max | 3.0 | 7.0 | 3.0 | 13.0 | 26.0 | 1.00 |

The LeREaD Labelizer achieves the **second-highest overall score** (19.0 / 0.79), just below WebAnno (21.0 / 0.81). It reaches the maximum technical (7.0) and data-format (3.0) subtotals: the points lost overall come from the publication category and from deliberately out-of-scope functional criteria (document-level labels, ontologies, biomedical integration, multi-user management).

## 8. Scope and limitations

The tool is intentionally narrow: it optimises for precise span-level legal citation annotation with hierarchical labels, co-reference clustering, and provenance separation in a no-install, privacy-preserving setting. It does not aim to provide collaborative project management, ontology services, document classification, or biomedical integrations. For teams needing those, a general platform remains the better choice; for the LeREaD scheme, those features were traded for schema configurability, full-length HTML fidelity, and accessibility for legal annotators.
