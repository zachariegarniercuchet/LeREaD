# AGENTS.md

Research codebase, not a package. No `requirements.txt`/`pyproject.toml`, no tests, no lint/CI. No PR process — work directly.

## Run from repo root

Scripts assume CWD is repo root (`Path.cwd()` is inserted into `sys.path`; `configs/config.py` defines `PROJECT_ROOT`). Run as modules from root:

```bash
python -m src.main_extraction --split dev --method DEC [...]
python -m src.main_coref --split dev --input-source gold [...]
python -m src.main_resolution --gold --split test --retriever bge-s
python src/evaluation/evaluate.py batch --batch-folder output/<run>/
```

## 3-stage pipeline (order matters)

1. **Extraction** `src/main_extraction.py`: raw HTML → labeled mentions. `--method AIO|DEC`, `--chunker paragraph|sentence`, `--fewshot-method greedy|random`, `--variant original|annotated`, `--data-dir` defaults to `DATA_DIR`. Output `output/`, per-file subfolders in split mode.
2. **Coref** `src/main_coref.py`: mentions → `docid` per mention. `--input-source gold|extraction` (`extraction` requires `--extraction-dir` + `--extraction-stage processed|final|0|1|2|3`). `--variant` defaults to `annotated` (unlike extraction). Output `output_coref/`.
3. **Resolution** `src/main_resolution.py`: profiles → `uri` via candidate pool. `--folder <stage-2-run>` under `--dir output_coref` OR `--gold --split <split>` (oracle; strips gold `uri`s first). `--retriever` must match the one used to build the index.

`--filename X` = single file (auto-scans `SPLITS`); `--split S` = whole folder. `--start/--end` (extraction only, 1-based inclusive) slices the file list.

## Precompute before running (order matters)

```bash
python -m src.precompute_chunks_cache --method paragraph   # sentence needs spacy en_core_web_trf
python -m src.precompute_pattern_dict                      # then:
python -m src.precompute_fewshot_examples --task extraction [--force]
python -m src.precompute_fewshot_examples --task coref --method random
python -m src.precompute_index --retriever bge-s           # -> cache/candidate_pool_index_<retriever>_v1/
python -m src.precompute_rpr                               # needed for coref fewshot
```

Fewshot cache filenames encode the weights active at precompute time: `examples_<greedy|random>_surf-<s>_struct-<t>.json`. Extraction requests per-method combos (`src/main_extraction.py:120`): AIO needs `surf-1.0_struct-1.0`, DEC0 needs `surf-0.0_struct-1.0`, DEC1–3 need `surf-1.0_struct-0.0` — but one precompute run writes only one file from the current `GREEDY_CONFIG` (`configs/config.py`). A full DEC run therefore needs three precompute passes with edited weights (see `cache/fewshot/`). Coref only supports `random` (`examples_coref_random.json`; `greedy` will FileNotFound). All precomputes skip cached work unless `--force`.

## Models, keys, local-only files

- `configs/constants.py:MODEL_MAPPING_NAME` maps CLI short names (`qwen7b`, `qwen32b`, `qwen38-27b`, `gpt-5.2`, `phi-4`, `saul-54b`, `muse`, `gemma4`). `gpt-5.2` goes through OpenAI API and requires `OPENAI_API_KEY`; anything else loads locally via `AssistantFactory.create(MODEL_MAPPING_NAME[...])`.
- `src/models/registry.py` is **gitignored** (see `.gitignore`) — open-weight runs fail without this local file. Do not recreate blindly; ask before inventing paths/quantization.
- Deps are undocumented; imports imply `torch`, `transformers`, `bitsandbytes`, `spacy` (+ `en_core_web_trf` for `--chunker sentence`), `beautifulsoup4`, `pandas`, `numpy`, `tqdm`, `openai`. No verified install command — check the environment first.

## Checkpoints and outputs

- Reruns resume silently: extraction (`<file>_processed/_0.._3/_final.html`), coref (`<file>_coref_mapping.json`), resolution (`*_resolution.html`). Use `--overwrite` (resolution), `--force` (precompute), `--with-timestamp` (new run folder) for fresh runs.
- `private/`, `output/`, `output_coref/`, `cache/`, `jobs/` are generated/local (most gitignored). Never edit or commit them; never read `private/` for task logic.
- Data: `data/original/<split>/` (raw) vs `data/annotated/<split>/` (gold). Valid splits in `configs/config.py:SPLITS` include `train|test|dev|incoming|extended_test|dev_train|no-gold`.

## Evaluation

`src/evaluation/evaluate.py single|batch|config` (gold at `data/annotated/{train,test,dev}/`). Filename matching is **exact-stem** (`src/evaluation/evaluate.py:190,235`): `foo_processed.html` will NOT pair with gold `foo.html` — strip the `_processed`/`_final` suffix or copy/rename first, despite what `EVALUATION_README.md` examples suggest. Batch subfolder mode requires one `*final.html` per doc subfolder (doc id = subfolder name before first `_`); it ignores `*_coref.html`/`*_resolution.html`. Coref/resolution outputs use `evaluate_coref.py` / `evaluate_resolution.py` instead. Batch auto-detects split from folder prefix (`test_...`); pass `--split` explicitly otherwise.
