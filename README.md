# LeREaD: A Dataset for Legal Reference Extraction and Disambiguation

## Overview

Legal documents are dense with references: a single court decision may cite dozens of statutes, prior judgments, and secondary materials. **LeREaD** is a benchmark for *legal reference extraction and disambiguation* on Canadian court and tribunal decisions from CanLII. It contains 100 decisions issued between 1982 and 2026, with 49,737 annotated spans (17,852 mentions and 31,885 components). Of these, 45 decisions (31,602 spans) are manually verified and mostly used for evaluation, and 55 are silver-annotated to provide additional training and validation data. Mentions are linked to reference sources with structured metadata, supporting both span-level extraction and document-level disambiguation.

## Tasks and Pipeline

Given a legal document, the benchmark covers three stages ([pipeline overview](img/pipeline_figure.pdf)):

1. **Reference extraction** — identify mentions in the text and assign each a mention type (*legislation*, *decision*, or *secondary source*), decomposed into nested components (*title*, *citation*, *fragment* for legislation/decisions; *title*, *authors*, *source*, *fragment* for secondary sources).
2. **Coreference resolution** — group the mentions within a document that denote the same source (e.g. a full citation and its later short forms) under a shared intra-document identifier (`docid`).
3. **`uri` linking** — map each group to a canonical entry in the reference repository through an inter-document identifier (`uri`).

Steps (2) and (3) together form **reference disambiguation**: determining which reference source each mention denotes. Each stage can be evaluated separately (`src/evaluation/`).

## Dataset

### Splits (`data/annotated/`, mirrored unannotated in `data/original/`)

| Split | Docs | Mentions | Components | Total spans |
|---|---|---|---|---|
| train | 4 | 2,720 | 5,672 | 8,392 |
| dev | 3 | 330 | 488 | 818 |
| test | 38 | 8,006 | 14,386 | 22,392 |
| silver | 55 | 6,796 | 11,339 | 18,135 |
| **Total** | **100** | **17,852** | **31,885** | **49,737** |

The 45 gold documents are split into train (4), dev (3), and test (38). The test set combines 4 fully manual documents (`test_manual/`) with 34 LLM-pre-annotated then human-verified documents (`test_assisted/`). Train and dev are small by design: they serve for demonstrations and model/prompt selection, while the large test split keeps results reliable across courts, tribunals, and mention types. The 55 `silver/` documents are automatically annotated (extraction + `docid`, with `uri`s mapped to 1,294 synthetic sources) and are intended as optional training/validation material only.

### Reference repository (candidate pools)

The 11,056 gold mentions resolve to 1,437 unique cited documents, released as two candidate pools for the linking stage:

- `data/gold_candidate_pool.csv` (6,437 entries): the 1,437 cited documents plus 5,000 never-cited distractors. Used for all reported results.
- `data/full_candidate_pool.csv` (7,731 entries): additionally includes the 1,294 synthetic silver sources, for training/validation with silver data.

(`data/cited_metadata.csv`, `data/extra_metadata.csv`, and `data/synthetic_metadata.csv` hold the underlying per-source metadata.)

## Repository Structure

```
LeREaD/
├── cache/                # Precomputed artifacts (chunk cache, few-shot examples,
│                         #   pattern dictionaries, candidate-pool indexes, RPR data).
│                         #   Large embedding matrices (*.npy) are excluded, rebuild
│                         #   them with src/precompute_index.py
├── configs/              # Splits, label scheme, model mapping, run configuration
├── data/
│   ├── annotated/        # train / dev / test (+ test_manual, test_assisted views) / silver
│   ├── original/         # Unannotated copies of the same documents
│   └── *.csv             # Gold / full candidate pools and source metadata
├── img/
│   ├── pipeline_figure.pdf          # Task overview figure
│   └── coverage_comparison_combined.png
├── scripts/              # Data preparation utilities (+ scripts/sync_supervisor.py,
│                         #   which maintains the review subset of this repo)
├── src/
│   ├── main_extraction.py  # Stage 1: copy-and-annotate LLM extraction (AIO / DEC prompting,
│   │                       #   paragraph- or sentence-aware chunking, pattern-coverage few-shot,
│   │                       #   protected-Levenshtein repair)
│   ├── main_coref.py       # Stage 2: incremental coreference with a growing
│   │                       #   Reference Profile Registry (-> docid)
│   ├── main_resolution.py  # Stage 3: dense/sparse retrieval linking (-> uri)
│   ├── precompute_*.py     # Cache builders (run once, in order, see below)
│   ├── evaluation/         # Stage-wise and end-to-end evaluation
│   └── prompts/            # Prompt templates and annotation guidelines pointer
├── Guide_annotation_V2_avril2026_Anonymized_for_review.pdf  # Annotation guidelines
└── README.md
```

## Running the Pipeline

Run from the repo root. Precompute caches once, in this order:

```bash
python -m src.precompute_chunks_cache --method paragraph
python -m src.precompute_pattern_dict
python -m src.precompute_fewshot_examples --task extraction
python -m src.precompute_fewshot_examples --task coref --method random
python -m src.precompute_index --retriever bge-s
python -m src.precompute_rpr
```

Then the three stages (here on `dev`/`test` with gold inputs; see `src/` docstrings and `configs/config.py` for full options):

```bash
python -m src.main_extraction --split dev --method DEC
python -m src.main_coref --split dev --input-source gold
python -m src.main_resolution --gold --split test --retriever bge-s
python src/evaluation/evaluate.py batch --batch-folder output/<run>/
```

Model short names map to local checkpoints or the OpenAI API in `configs/constants.py` (`MODEL_MAPPING_NAME`).

## Annotation

The gold documents were annotated by two legal experts with `LeREaDLabelizer`, a web-based hierarchical span annotation tool, in two phases: 11 fully manual documents, then 34 documents pre-annotated by the extraction module and verified/corrected by the annotators (final pilot agreement: span micro-F1 0.91; verification raised throughput by 29% at equal quality). The annotation guidelines are released in this repo (`Guide_annotation_V2_avril2026_Anonymized_for_review.pdf`).

## Acknowledgments

### Funding & Resources

This project was supported by Compute Canada, which provided computational resources essential for large-scale model benchmarking.

### AI-Assisted Development

This codebase has been developed with the assistance of AI tools including but not limited to:
- ChatGPT (OpenAI)
- Claude (Anthropic)
- GitHub Copilot (Microsoft)

In accordance with ACL guidelines, we acknowledge the use of code assistance tools during the development of this project. All generated code was reviewed for correctness and compliance with licensing requirements.

## Citation

[Citation information to be added upon publication]

## License

This work is licensed under the **Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)** license.

Under this license, you are free to:
- Share and adapt the material for non-commercial purposes
- Provide appropriate attribution to the original authors

You are not permitted to:
- Use this material for commercial purposes

For more details, see [CC BY-NC 4.0 License](https://creativecommons.org/licenses/by-nc/4.0/)

---

**Note**: This is an anonymized version of the LeREaD repository prepared for the ACL review process.
