"""Cross-encoder (pair-classification) student for LeREaD coreference.

Why a separate module: coreference is not generative here. For each mention we
score it against every existing cluster profile and take the best match above a
threshold, or open a new cluster. DeBERTa-v3-large is the backbone so the coref
step shares one encoder family with the extraction step.

Pure-python helpers come first (no torch import) so `--dry-run` works anywhere.
"""
from __future__ import annotations

import json
import math
import random
from pathlib import Path

import numpy as np


# --------------------------------------------------------------------------
# Label / text helpers
# --------------------------------------------------------------------------
def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def pairwise_scores_to_probs(logits: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid over an array of logits."""
    logits = np.asarray(logits, dtype=np.float64)
    return 1.0 / (1.0 + np.exp(-np.clip(logits, -500.0, 500.0)))


# --------------------------------------------------------------------------
# Pair tokenization -> records (no torch at import time)
# --------------------------------------------------------------------------
def build_records(pairs, tok, max_len=512):
    """examples -> window records for cross-encoder training.

    Each pair is a dict with keys input_a / input_b / label. We tokenize the
    (input_a, input_b) pair so DeBERTa inserts its [SEP] between mention and
    profile. Returns (records, n_empty) where each record is
    {input_ids, attention_mask, label}.
    """
    recs, n_empty = [], 0
    for ex in pairs:
        a = ex.get("input_a", "") or ""
        b = ex.get("input_b", "") or ""
        if not a and not b:
            n_empty += 1
            continue
        enc = tok(
            a, b,
            truncation=True,
            max_length=max_len,
            return_tensors=None,
        )
        recs.append({
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "label": float(ex.get("label", 0.0)),
        })
    return recs, n_empty


# --------------------------------------------------------------------------
# Model (torch imported lazily)
# --------------------------------------------------------------------------
def build_model(config_or_model_path, pretrained=True, dropout=0.1,
                trust_remote_code=False, gradient_checkpointing=True):
    """DeBERTa encoder + a single binary classification head on [CLS]."""
    import torch.nn as nn
    from transformers import AutoConfig, AutoModel

    class CorefCrossEncoder(nn.Module):
        def __init__(self):
            super().__init__()
            if pretrained:
                self.encoder = AutoModel.from_pretrained(
                    config_or_model_path, trust_remote_code=trust_remote_code)
            else:
                cfg = AutoConfig.from_pretrained(
                    config_or_model_path, trust_remote_code=trust_remote_code)
                self.encoder = AutoModel.from_config(cfg, trust_remote_code=trust_remote_code)
            # Keep fp32 parameters/optimizer masters; autocast reduces activation
            # memory without making AdamW unstable.
            self.encoder.float()
            if gradient_checkpointing:
                enable_kwargs = {"use_reentrant": False}
                try:
                    self.encoder.gradient_checkpointing_enable(
                        gradient_checkpointing_kwargs=enable_kwargs
                    )
                except TypeError:
                    self.encoder.gradient_checkpointing_enable()
                self.encoder.config.use_cache = False
            h = self.encoder.config.hidden_size
            self.drop = nn.Dropout(dropout)
            self.classifier = nn.Linear(h, 1)

        def forward(self, input_ids, attention_mask, labels=None):
            hs = self.drop(self.encoder(
                input_ids=input_ids, attention_mask=attention_mask
            ).last_hidden_state)
            cls_repr = hs[:, 0, :]                      # [CLS] representation
            logits = self.classifier(cls_repr).squeeze(-1)
            out = {"logits": logits}
            if labels is not None:
                bce = nn.BCEWithLogitsLoss()
                out["loss"] = bce(logits, labels.float())
            return out

    return CorefCrossEncoder()


def _collate(batch, pad_id):
    import torch

    L = max(len(b["input_ids"]) for b in batch)

    def pad(key, val):
        return torch.tensor([b[key] + [val] * (L - len(b[key])) for b in batch])

    labels = torch.tensor([b["label"] for b in batch], dtype=torch.float32)
    return {
        "input_ids": pad("input_ids", pad_id),
        "attention_mask": pad("attention_mask", 0),
        "labels": labels,
    }


def _candidate_groups(examples):
    """Group serialized candidates by one mention for listwise training."""
    groups = {}
    for ex in examples:
        key = (ex.get("source_file"), ex.get("mention_id"))
        groups.setdefault(key, []).append(ex)
    return [
        group for group in groups.values()
        if len(group) > 1
        and any(float(ex.get("label", 0.0)) >= 0.5 for ex in group)
    ]


def _group_loss(model, tokenizer, group, device, max_len, candidate_batch_size=8):
    import torch

    positive = [ex for ex in group if float(ex.get("label", 0.0)) >= 0.5]
    if len(positive) != 1:
        raise ValueError(
            "Each listwise coreference group must contain exactly one positive "
            f"candidate; got {len(positive)} for {group[0].get('mention_id')!r}."
        )
    negatives = [ex for ex in group if ex is not positive[0]]
    chunks = [
        [positive[0]] + negatives[i:i + candidate_batch_size - 1]
        for i in range(0, len(negatives), max(1, candidate_batch_size - 1))
    ]
    losses = []
    for chunk in chunks:
        input_a = [ex.get("input_a", "") or "" for ex in chunk]
        input_b = [ex.get("input_b", "") or "" for ex in chunk]
        enc = tokenizer(
            input_a, input_b, truncation=True, max_length=max_len,
            padding=True, return_tensors="pt",
        )
        ids = enc["input_ids"].to(device)
        att = enc["attention_mask"].to(device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16,
                            enabled=(device.type == "cuda")):
            logits = model(ids, att)["logits"]
            losses.append(torch.nn.functional.cross_entropy(
                logits.unsqueeze(0), torch.zeros(1, dtype=torch.long, device=device)
            ))
    return torch.stack(losses).mean()


def predict_scores(model, tok, pairs, device, max_len=512, batch_size=16):
    """Score each (input_a, input_b) pair -> list of match probabilities in [0,1]."""
    import torch

    model.eval()
    scores = []
    for i in range(0, len(pairs), batch_size):
        batch = pairs[i:i + batch_size]
        text_a = [p.get("input_a", "") or "" for p in batch]
        text_b = [p.get("input_b", "") or "" for p in batch]
        enc = tok(
            text_a, text_b,
            truncation=True, max_length=max_len,
            padding=True, return_tensors="pt",
        )
        ids = enc["input_ids"].to(device)
        att = enc["attention_mask"].to(device)
        with torch.no_grad(), torch.autocast(
            device_type="cuda", dtype=torch.bfloat16,
            enabled=(device.type == "cuda"),
        ):
            logits = model(ids, att)["logits"]
        # CUDA autocast returns bfloat16 logits; NumPy does not support
        # converting that dtype directly.
        probs = torch.sigmoid(logits.float()).cpu().numpy().tolist()
        scores.extend(probs)
    return scores


def evaluate_pairs(model, tok, pairs, device, max_len=512):
    """Score dev pairs and report precision/recall/F1 across thresholds.

    Returns the threshold maximizing F1 and a dict of per-threshold metrics.
    """
    probs = predict_scores(model, tok, pairs, device, max_len)
    labels = np.array([float(p.get("label", 0.0)) for p in pairs])

    thresholds = np.linspace(0.05, 0.95, 19)
    results = []
    best = {"threshold": 0.5, "f1": -1.0, "precision": 0.0, "recall": 0.0}
    for t in thresholds:
        preds = (np.array(probs) >= t).astype(int)
        y = labels.astype(int)
        tp = int(((preds == 1) & (y == 1)).sum())
        fp = int(((preds == 1) & (y == 0)).sum())
        fn = int(((preds == 0) & (y == 1)).sum())
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        results.append({"threshold": float(t), "precision": precision,
                        "recall": recall, "f1": f1})
        if f1 > best["f1"]:
            best = {"threshold": float(t), "f1": f1,
                    "precision": precision, "recall": recall}
    return best, results


# --------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------
def train_coref_encoder(model_short, train_on, train_examples, dev_examples=None,
                        output_root=None, model_path=None, max_len=512,
                        lr=2e-5, head_lr=1e-3, epochs=5, batch_size=8,
                        grad_accum=2, warmup_ratio=0.1, weight_decay=0.01,
                        seed=42, max_hard_negatives=0, dev_split="dev",
                        candidate_batch_size=8, gradient_checkpointing=True):
    import torch
    from torch.utils.data import DataLoader
    from transformers import AutoTokenizer, get_linear_schedule_with_warmup
    from tqdm import tqdm

    from src.training.lora import resolve_model_path

    trust = False
    if model_path is None:
        model_path, entry = resolve_model_path(model_short)
        trust = bool(entry.get("trust_remote_code", False))

    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tok = AutoTokenizer.from_pretrained(model_path, use_fast=True, trust_remote_code=trust)
    assert tok.is_fast, "A fast tokenizer is required."

    recs, n_empty = build_records(train_examples, tok, max_len)
    groups = _candidate_groups(train_examples)
    print(f"[coref-encoder] {len(train_examples)} pairs -> {len(recs)} records "
          f"({n_empty} empty pairs skipped)")
    print(f"[coref-encoder] listwise groups={len(groups)}")

    run_name = (f"{model_short}_coref_{train_on}_lr{lr:g}_hlr{head_lr:g}"
                f"_ep{epochs}_hn{max_hard_negatives}"
                + ("" if seed == 42 else f"_s{seed}"))
    out_dir = Path(output_root or default_output_root()) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    model = build_model(
        model_path,
        pretrained=True,
        trust_remote_code=trust,
        gradient_checkpointing=gradient_checkpointing,
    ).to(device)
    head = list(model.classifier.parameters())
    opt = torch.optim.AdamW(
        [{"params": model.encoder.parameters(), "lr": lr},
         {"params": head, "lr": head_lr}], weight_decay=weight_decay)

    g = torch.Generator().manual_seed(seed)
    # A loader item is one mention's candidate list. Candidate lists can be
    # much larger than batch_size, so they are chunked separately below.
    loader = DataLoader(groups, batch_size=1, shuffle=True, generator=g,
                        collate_fn=lambda b: b)
    total = math.ceil(len(loader) / grad_accum) * epochs
    sched = get_linear_schedule_with_warmup(opt, round(warmup_ratio * total), total)
    print(
        f"[coref-encoder] device={device} epochs={epochs} "
        f"batches/epoch={len(loader)} optimizer_steps={total} "
        f"candidate_batch_size={candidate_batch_size} grad_accum={grad_accum}"
    )

    best, history = -1.0, []
    best_threshold = 0.5
    for ep in range(1, epochs + 1):
        model.train()
        run_loss = 0.0
        progress = tqdm(
            enumerate(loader),
            total=len(loader),
            desc=f"[coref-encoder] epoch {ep}/{epochs}",
            unit="batch",
        )
        for step, batch in progress:
            group = batch[0]
            loss = _group_loss(
                model, tok, group, device, max_len, candidate_batch_size
            ) / grad_accum
            if not torch.isfinite(loss):
                raise RuntimeError(
                    f"Non-finite loss at epoch {ep}, step {step} "
                    f"(encoder dtype={next(model.encoder.parameters()).dtype}).")
            loss.backward()
            run_loss += loss.item() * grad_accum
            if (step + 1) % grad_accum == 0 or step + 1 == len(loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step(); sched.step(); opt.zero_grad()
            progress.set_postfix(loss=f"{run_loss / (step + 1):.4f}")
        progress.close()
        train_loss = run_loss / len(loader)
        print(f"[coref-encoder] epoch {ep}/{epochs} training complete "
              f"loss={train_loss:.4f}")
        msg = f"[coref-encoder] epoch {ep}/{epochs} loss={train_loss:.4f}"
        score = None
        if dev_examples:
            print(f"[coref-encoder] evaluating {len(dev_examples)} dev pairs...")
            best_t, _ = evaluate_pairs(model, tok, dev_examples, device, max_len)
            score = best_t["f1"]
            best_threshold = best_t["threshold"]
            msg += (f" | dev F1={score:.3f} @ threshold={best_threshold:.2f} "
                    f"(P={best_t['precision']:.3f} R={best_t['recall']:.3f})")
        print(msg)
        sequential = None
        if dev_split:
            sequential = evaluate_sequential_split(
                model, tok, device, dev_split,
                max_len=max_len, batch_size=batch_size,
                thresholds=np.linspace(0.05, 0.95, 19),
            )
            best_threshold = sequential["threshold"]
            print(
                f"[coref-encoder] sequential {dev_split}: "
                f"CoNLL={sequential['conll_f1']:.3f} "
                f"LEA={sequential['lea_f1']:.3f} "
                f"threshold={best_threshold:.2f}"
            )
        selection = sequential["conll_f1"] if sequential else score
        history.append({
            "epoch": ep, "loss": train_loss, "dev_pair_f1": score,
            "dev_sequential": sequential, "threshold": best_threshold,
        })
        if selection is None or selection > best:
            best = selection if selection is not None else best
            torch.save(model.state_dict(), out_dir / "model.pt")
            model.encoder.config.save_pretrained(out_dir)
            tok.save_pretrained(out_dir)
            with (out_dir / "run_config.json").open("w") as f:
                json.dump({
                    "model_short": model_short, "train_on": train_on,
                    "base_path": model_path, "max_len": max_len,
                    "lr": lr, "head_lr": head_lr, "epochs": epochs,
                    "max_hard_negatives": max_hard_negatives,
                    "best_epoch": ep, "best_dev_f1": score,
                    "best_sequential_conll_f1": (
                        sequential["conll_f1"] if sequential else None
                    ),
                    "best_sequential_lea_f1": (
                        sequential["lea_f1"] if sequential else None
                    ),
                    "best_threshold": best_threshold,
                    "seed": seed,
                    "dev_split": dev_split,
                    "objective": "listwise_candidate_cross_entropy",
                    "gradient_checkpointing": gradient_checkpointing,
                    "history": history,
                }, f, indent=2)
    print(f"[coref-encoder] saved -> {out_dir}")
    return out_dir


def evaluate_sequential_split(model, tok, device, split, max_len=512,
                              batch_size=16, thresholds=None):
    """Evaluate the exact document-order resolver and tune its threshold."""
    from src.ann_extractor import extract_parent_level_annotations
    from src.evaluation.evaluation_coref import (
        aggregate_raw, evaluate_coref_raw, finalize_raw,
    )
    from src.training.infer_coref_encoder import CorefEncoderRuntime
    from src.main_coref import process_document_coref
    from configs.config import DATA_DIR

    thresholds = list(thresholds if thresholds is not None
                      else np.linspace(0.05, 0.95, 19))
    paths = sorted((Path(DATA_DIR) / "annotated" / split).glob("*.html"))
    documents = []
    for path in paths:
        html = path.read_text(encoding="utf-8")
        mentions = extract_parent_level_annotations(html)
        gold_mapping = {
            m.html_tag.attributes["id"]: m.html_tag.attributes.get("docid")
            for m in mentions
        }
        documents.append((html, gold_mapping))

    best = None
    for threshold in thresholds:
        raw_scores = []
        for html, gold_mapping in documents:
            mentions = extract_parent_level_annotations(html)
            runtime = CorefEncoderRuntime(
                model, tok, device, float(threshold), max_len, batch_size
            )
            predicted, _ = process_document_coref(
                html, mentions, "", [], runtime, 10,
                ["docid", "alternative_titles", "citations",
                 "fragments_mentioned", "authors"],
            )
            gold_clusters = _mapping_to_clusters(gold_mapping)
            pred_clusters = _mapping_to_clusters(predicted)
            raw_scores.append(evaluate_coref_raw(gold_clusters, pred_clusters))
        scores = finalize_raw(aggregate_raw(raw_scores))
        candidate = {
            "threshold": float(threshold),
            "conll_f1": scores["CoNLL"]["f1"],
            "lea_f1": scores["LEA"]["f1"],
            "metrics": scores,
        }
        if (best is None or candidate["conll_f1"] > best["conll_f1"]
                or (candidate["conll_f1"] == best["conll_f1"]
                    and candidate["lea_f1"] > best["lea_f1"])):
            best = candidate
    return best or {
        "threshold": 0.5, "conll_f1": 0.0, "lea_f1": 0.0,
        "metrics": finalize_raw(aggregate_raw([])),
    }


def _mapping_to_clusters(mapping):
    clusters = {}
    for mention_id, docid in mapping.items():
        clusters.setdefault(docid, []).append(mention_id)
    return list(clusters.values())


def default_output_root() -> str:
    try:
        from src.models.registry import MODELS_DIR
        return str(Path(MODELS_DIR) / "output_training")
    except ImportError:
        return "output_training"


def load_coref_encoder(path, device=None):
    """Load a saved cross-encoder run folder -> (model, tokenizer, run_config)."""
    import torch
    from transformers import AutoTokenizer

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = json.loads((Path(path) / "run_config.json").read_text())
    tok = AutoTokenizer.from_pretrained(path, use_fast=True)
    model = build_model(str(path), pretrained=False, gradient_checkpointing=False)
    model.load_state_dict(torch.load(Path(path) / "model.pt", map_location="cpu"))
    return model.to(device).eval(), tok, cfg
