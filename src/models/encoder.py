"""Encoder (token-classification) student for LeREaD extraction.

Why a separate module: the LLM route is copy-and-annotate (text -> tagged text);
an encoder predicts a label per sub-word instead. We bridge the two so nothing
else in the repo changes:

  training data : the SAME (input, output) pairs as the AIO LoRA run
                  (src.training.dataset.get_or_build_dataset), where `output`
                  is the chunk with <decision><title>..</title></decision> tags.
                  We parse the tags back into character spans.
  model         : one encoder + TWO BIO heads (mention head: 3 types,
                  component head: 5 types) -> handles the nesting without
                  nested tagging.
  inference     : predict spans, enforce hierarchy, and RENDER the tagged
                  string again, so the output goes through the existing
                  verification / post-processing / evaluate.py unchanged.

Pure-python helpers come first (no torch import) so `--dry-run` works anywhere.
"""
from __future__ import annotations

import bisect
import json
import math
import random
import re
from pathlib import Path

import numpy as np

# --------------------------------------------------------------------------
# Label inventory
# --------------------------------------------------------------------------
MENTION_LABELS = ["decision", "legislation", "secondary sources"]
COMPONENT_LABELS = ["title", "citation", "fragment", "authors", "source"]
N_MENTION = 2 * len(MENTION_LABELS) + 1      # O + B/I per type
N_COMPONENT = 2 * len(COMPONENT_LABELS) + 1

ALLOWED_COMPONENTS = {
    "decision": {"title", "citation", "fragment"},
    "legislation": {"title", "citation", "fragment"},
    "secondary sources": {"title", "authors", "source", "fragment"},
}

# Tag spelling used when RENDERING. Must match what your AIO dataset emits
# (check one line of data/cache/training/aio_*.json). Change here if it is
# e.g. <secondary_sources> instead of <secondary sources>.
TAG_NAME = {n: n for n in MENTION_LABELS + COMPONENT_LABELS}

TAG_RE = re.compile(
    r"<(/?)(decision|legislation|secondary[ _]sources|title|citation|fragment|authors|source)>"
)


def default_output_root() -> str:
    """<MODELS_DIR>/output_training (scratch on Tamia), local fallback."""
    try:
        from src.models.registry import MODELS_DIR
        return str(Path(MODELS_DIR) / "output_training")
    except ImportError:
        return "output_training"


# --------------------------------------------------------------------------
# Tagged text <-> character spans
# --------------------------------------------------------------------------
def parse_tagged(tagged: str) -> tuple[str, list[tuple[int, int, str]]]:
    """Strip tags -> (plain_text, [(start, end, label)]) in plain-text offsets."""
    parts, spans, stack = [], [], []
    pos = last = 0
    for m in TAG_RE.finditer(tagged):
        chunk = tagged[last:m.start()]
        parts.append(chunk)
        pos += len(chunk)
        last = m.end()
        name = m.group(2).replace("_", " ")
        if not m.group(1):
            stack.append((name, pos))
        else:
            for j in range(len(stack) - 1, -1, -1):
                if stack[j][0] == name:
                    _, s = stack.pop(j)
                    spans.append((s, pos, name))
                    break
    parts.append(tagged[last:])
    return "".join(parts), spans


def _trim(text: str, s: int, e: int) -> tuple[int, int]:
    while s < e and text[s].isspace():
        s += 1
    while e > s and text[e - 1].isspace():
        e -= 1
    return s, e


def assemble(mentions, comps):
    """Enforce the hierarchy: non-overlapping mentions, components inside them,
    only components allowed for the mention type, no overlapping components."""
    kept, end = [], -1
    for s, e, n in sorted(mentions, key=lambda x: (x[0], -x[1])):
        if s >= end:
            kept.append([s, e, n, []])
            end = e
    for cs, ce, cn in sorted(comps):
        for m in kept:
            if m[0] <= cs < m[1]:
                if cn in ALLOWED_COMPONENTS[m[2]] and (not m[3] or cs >= m[3][-1][1]):
                    m[3].append((cs, min(ce, m[1]), cn))
                break
    return kept


def render(text: str, kept) -> str:
    """Insert tags into `text` (verbatim copy) -> same format the LLM emits."""
    out, pos = [], 0
    for ms, me, mn, comps in kept:
        out += [text[pos:ms], f"<{TAG_NAME[mn]}>"]
        p = ms
        for cs, ce, cn in comps:
            t = TAG_NAME[cn]
            out += [text[p:cs], f"<{t}>{text[cs:ce]}</{t}>"]
            p = ce
        out += [text[p:me], f"</{TAG_NAME[mn]}>"]
        pos = me
    out.append(text[pos:])
    return "".join(out)


# --------------------------------------------------------------------------
# Windows, BIO labels, decoding
# --------------------------------------------------------------------------
def _windows(tok, text: str, max_len: int, stride: int):
    """Split into overlapping sub-word windows -> [(input_ids, offsets)]."""
    enc = tok(text, return_offsets_mapping=True, truncation=True, max_length=max_len,
              stride=stride, return_overflowing_tokens=True)
    return [(ids, [tuple(o) for o in offs])
            for ids, offs in zip(enc["input_ids"], enc["offset_mapping"])]


def _bio(offsets, spans, label_list):
    """Token labels: 0=O, 1+2k=B-k, 2+2k=I-k, -100=special token."""
    out = [-100 if a == b else 0 for a, b in offsets]
    real = [i for i, (a, b) in enumerate(offsets) if a != b]
    ends = [offsets[i][1] for i in real]
    for s, e, name in spans:
        if name not in label_list:
            continue
        k = label_list.index(name)
        j = bisect.bisect_right(ends, s)
        while j < len(real) and offsets[real[j]][0] < e:
            i = real[j]
            a, b = offsets[i]
            out[i] = 1 + 2 * k if a <= s < b else 2 + 2 * k
            j += 1
    return out


def _aggregate(windows):
    """Average per-token probabilities over overlapping windows."""
    acc = {}
    for offs, pm, pc in windows:
        for i, key in enumerate(offs):
            if key[0] == key[1]:
                continue
            if key in acc:
                acc[key][0] += pm[i]; acc[key][1] += pc[i]; acc[key][2] += 1
            else:
                acc[key] = [pm[i].copy(), pc[i].copy(), 1]
    keys = sorted(acc)
    return (keys, [acc[k][0] / acc[k][2] for k in keys],
            [acc[k][1] / acc[k][2] for k in keys])


def _decode(keys, probs, labels, text):
    spans, cur = [], None
    for (a, b), p in zip(keys, probs):
        l = int(np.argmax(p))
        if l == 0:
            if cur:
                spans.append(cur); cur = None
            continue
        k, is_b = (l - 1) // 2, (l - 1) % 2 == 0
        if cur is not None and not is_b and cur[2] == k:
            cur[1] = b
        else:
            if cur:
                spans.append(cur)
            cur = [a, b, k]
    if cur:
        spans.append(cur)
    out = []
    for a, b, k in spans:
        s, e = _trim(text, a, b)
        if e > s:
            out.append((s, e, labels[k]))
    return out


def _spans_to_kept(keys, pm, pc, text):
    return assemble(_decode(keys, pm, MENTION_LABELS, text),
                    _decode(keys, pc, COMPONENT_LABELS, text))


# --------------------------------------------------------------------------
# Scoring (exact boundary + type, micro-F1, same criterion as the paper)
# --------------------------------------------------------------------------
class _Scorer:
    def __init__(self):
        self.c = {"mention": [0, 0, 0], "component": [0, 0, 0]}   # tp, n_pred, n_gold

    def add(self, doc_id, gold_spans, kept):
        pm = {(s, e, n) for s, e, n, _ in kept}
        pc = {(s, e, n) for _, _, _, cs in kept for s, e, n in cs}
        gm = {sp for sp in gold_spans if sp[2] in MENTION_LABELS}
        gc = {sp for sp in gold_spans if sp[2] in COMPONENT_LABELS}
        for key, p, g in (("mention", pm, gm), ("component", pc, gc)):
            self.c[key][0] += len(p & g); self.c[key][1] += len(p); self.c[key][2] += len(g)

    @staticmethod
    def _f1(tp, p, g):
        pr, rc = tp / p if p else 0.0, tp / g if g else 0.0
        return 2 * pr * rc / (pr + rc) if pr + rc else 0.0

    def result(self):
        r = {k: self._f1(*v) for k, v in self.c.items()}
        tp, p, g = (sum(v[i] for v in self.c.values()) for i in range(3))
        r["overall"] = self._f1(tp, p, g)
        return r


def tokenization_ceiling(examples, tok, max_len=512, stride=128, show=0) -> dict:
    """F1 of gold labels pushed through tokenize -> BIO -> decode -> assemble.
    This is the best exact-match F1 this tokenizer can ever reach."""
    sc = _Scorer()
    for i, ex in enumerate(examples):
        text, spans = parse_tagged(ex["output"])
        wins = []
        for _, offs in _windows(tok, text, max_len, stride):
            lm = np.maximum(_bio(offs, spans, MENTION_LABELS), 0)
            lc = np.maximum(_bio(offs, spans, COMPONENT_LABELS), 0)
            wins.append((offs, np.eye(N_MENTION)[lm], np.eye(N_COMPONENT)[lc]))
        kept = _spans_to_kept(*_aggregate(wins), text)
        sc.add(i, spans, kept)
        if show:
            pred = {(s, e, n) for s, e, n, _ in kept} | {(s, e, n) for *_, cs in kept for s, e, n in cs}
            for s, e, n in [sp for sp in spans if sp not in pred][:show]:
                print(f"  [ceiling miss] doc{i} {n} {text[s:e]!r} ...context {text[max(0, s-20):e+20]!r}")
    return sc.result()


def build_records(examples, tok, max_len=512, stride=128):
    """examples -> window records for training. Returns (records, n_text_mismatch)."""
    recs, mismatch = [], 0
    for ex in examples:
        text, spans = parse_tagged(ex["output"])
        mismatch += int(text != ex["input"])
        for ids, offs in _windows(tok, text, max_len, stride):
            recs.append({
                "input_ids": ids,
                "attention_mask": [1] * len(ids),
                "mention_labels": _bio(offs, spans, MENTION_LABELS),
                "component_labels": _bio(offs, spans, COMPONENT_LABELS),
            })
    return recs, mismatch


# --------------------------------------------------------------------------
# Model (torch imported lazily)
# --------------------------------------------------------------------------
def build_model(config_or_model_path, pretrained=True, trust_remote_code=False, dropout=0.1):
    import torch.nn as nn
    from transformers import AutoConfig, AutoModel

    class TwoHeadTagger(nn.Module):
        def __init__(self):
            super().__init__()
            if pretrained:
                self.encoder = AutoModel.from_pretrained(
                    config_or_model_path, trust_remote_code=trust_remote_code)
            else:
                cfg = AutoConfig.from_pretrained(
                    config_or_model_path, trust_remote_code=trust_remote_code)
                self.encoder = AutoModel.from_config(cfg, trust_remote_code=trust_remote_code)
            # transformers>=5 loads the checkpoint dtype (deberta-v3 ships fp16) ->
            # AdamW on fp16 params gives NaN. Keep fp32 masters; autocast does bf16.
            self.encoder.float()
            h = self.encoder.config.hidden_size
            self.drop = nn.Dropout(dropout)
            self.mention_head = nn.Linear(h, N_MENTION)
            self.component_head = nn.Linear(h, N_COMPONENT)

        def forward(self, input_ids, attention_mask, mention_labels=None, component_labels=None):
            hs = self.drop(self.encoder(input_ids=input_ids,
                                        attention_mask=attention_mask).last_hidden_state)
            ml, cl = self.mention_head(hs).float(), self.component_head(hs).float()
            out = {"mention_logits": ml, "component_logits": cl}
            if mention_labels is not None:
                ce = nn.CrossEntropyLoss(ignore_index=-100)
                out["loss"] = (ce(ml.view(-1, ml.size(-1)), mention_labels.view(-1))
                               + ce(cl.view(-1, cl.size(-1)), component_labels.view(-1)))
            return out

    return TwoHeadTagger()


def _collate(batch, pad_id):
    import torch

    L = max(len(b["input_ids"]) for b in batch)

    def pad(key, val):
        return torch.tensor([b[key] + [val] * (L - len(b[key])) for b in batch])

    return {"input_ids": pad("input_ids", pad_id), "attention_mask": pad("attention_mask", 0),
            "mention_labels": pad("mention_labels", -100),
            "component_labels": pad("component_labels", -100)}


def predict_kept(model, tok, text, device, max_len=512, stride=128, batch_size=16):
    """text -> hierarchical spans [[s, e, mention_type, [(cs, ce, comp_type)]]]."""
    import torch

    wins = _windows(tok, text, max_len, stride)
    results = []
    model.eval()
    for i in range(0, len(wins), batch_size):
        batch = wins[i:i + batch_size]
        L = max(len(w[0]) for w in batch)
        ids = torch.full((len(batch), L), tok.pad_token_id, dtype=torch.long)
        att = torch.zeros((len(batch), L), dtype=torch.long)
        for j, (w, _) in enumerate(batch):
            ids[j, :len(w)] = torch.tensor(w); att[j, :len(w)] = 1
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16,
                                             enabled=(device.type == "cuda")):
            out = model(ids.to(device), att.to(device))
        pm = torch.softmax(out["mention_logits"], -1).cpu().numpy()
        pc = torch.softmax(out["component_logits"], -1).cpu().numpy()
        for j, (_, offs) in enumerate(batch):
            results.append((offs, pm[j, :len(offs)], pc[j, :len(offs)]))
    return _spans_to_kept(*_aggregate(results), text)


def evaluate_examples(model, tok, examples, device, max_len=512, stride=128) -> dict:
    sc = _Scorer()
    for i, ex in enumerate(examples):
        text, spans = parse_tagged(ex["output"])
        sc.add(i, spans, predict_kept(model, tok, text, device, max_len, stride))
    return sc.result()


def diagnose(model, tok, examples, device, max_len=512, stride=128, name="", show=6):
    """Per-label exact vs overlap F1 + a few gold spans the model got wrong."""
    per, shown = {}, 0
    for ex in examples:
        text, gold = parse_tagged(ex["output"])
        kept = predict_kept(model, tok, text, device, max_len, stride)
        pred = [(s, e, n) for s, e, n, _ in kept] + [(s, e, n) for *_, cs in kept for s, e, n in cs]
        pset = set(pred)
        for n in {x[2] for x in gold} | {x[2] for x in pred}:
            g = [x for x in gold if x[2] == n]
            p = [x for x in pred if x[2] == n]
            used, ov = set(), 0
            for gs, ge, _ in g:
                for j, (ps, pe, _) in enumerate(p):
                    if j not in used and ps < ge and pe > gs:
                        used.add(j); ov += 1; break
            d = per.setdefault(n, [0, 0, 0, 0])
            d[0] += len(set(g) & set(p)); d[1] += len(p); d[2] += len(g); d[3] += ov
        for gs, ge, gn in gold:
            if shown < show and (gs, ge, gn) not in pset:
                near = [text[a:b] for a, b, nn in pred if a < ge and b > gs and nn == gn]
                print(f"  gold {gn}: {text[gs:ge]!r}  ->  pred overlap: {near!r}")
                shown += 1
    print(f"[diagnose {name}] label | exact F1 | overlap F1 | n_pred | n_gold")
    tot = [0, 0, 0, 0]
    for n, (tp, p, g, ov) in sorted(per.items()):
        print(f"  {n:18s} {2*tp/max(p+g,1):.3f}  {2*ov/max(p+g,1):.3f}  {p:6d} {g:6d}")
        tot = [a + b for a, b in zip(tot, (tp, p, g, ov))]
    print(f"  {'ALL':18s} {2*tot[0]/max(tot[1]+tot[2],1):.3f}  {2*tot[3]/max(tot[1]+tot[2],1):.3f}")


# --------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------
def train_encoder(model_short, train_on, train_examples, dev_examples=None,
                  output_root=None, model_path=None, max_len=512, stride=128,
                  lr=2e-5, head_lr=1e-3, epochs=8, batch_size=8, grad_accum=2,
                  warmup_ratio=0.1, weight_decay=0.01, seed=42):
    import torch
    from torch.utils.data import DataLoader
    from transformers import AutoTokenizer, get_linear_schedule_with_warmup

    from src.training.lora import resolve_model_path

    trust = False
    if model_path is None:
        model_path, entry = resolve_model_path(model_short)
        trust = bool(entry.get("trust_remote_code", False))

    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tok = AutoTokenizer.from_pretrained(model_path, use_fast=True, trust_remote_code=trust)
    assert tok.is_fast, "A fast tokenizer is required for offset mappings."

    recs, mism = build_records(train_examples, tok, max_len, stride)
    print(f"[encoder] {len(train_examples)} chunks -> {len(recs)} windows "
          f"({mism} chunks where input != tag-stripped output)")

    run_name = f"{model_short}_{train_on}_lr{lr:g}_hlr{head_lr:g}_ep{epochs}"
    out_dir = Path(output_root or default_output_root()) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    model = build_model(model_path, pretrained=True, trust_remote_code=trust).to(device)
    heads = list(model.mention_head.parameters()) + list(model.component_head.parameters())
    opt = torch.optim.AdamW(
        [{"params": model.encoder.parameters(), "lr": lr},
         {"params": heads, "lr": head_lr}], weight_decay=weight_decay)

    g = torch.Generator().manual_seed(seed)
    loader = DataLoader(recs, batch_size=batch_size, shuffle=True, generator=g,
                        collate_fn=lambda b: _collate(b, tok.pad_token_id))
    total = math.ceil(len(loader) / grad_accum) * epochs
    sched = get_linear_schedule_with_warmup(opt, round(warmup_ratio * total), total)

    best, history = -1.0, []
    for ep in range(1, epochs + 1):
        model.train()
        run_loss = 0.0
        for step, batch in enumerate(loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16,
                                enabled=(device.type == "cuda")):
                loss = model(**batch)["loss"] / grad_accum
            if not torch.isfinite(loss):
                raise RuntimeError(
                    f"Non-finite loss at epoch {ep}, step {step} "
                    f"(encoder dtype={next(model.encoder.parameters()).dtype}). "
                    "Check dtype / learning rate before continuing.")
            loss.backward()
            run_loss += loss.item() * grad_accum
            if (step + 1) % grad_accum == 0 or step + 1 == len(loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step(); sched.step(); opt.zero_grad()
        msg = f"[encoder] epoch {ep}/{epochs} loss={run_loss / len(loader):.4f}"
        score = None
        if dev_examples:
            res = evaluate_examples(model, tok, dev_examples, device, max_len, stride)
            score = res["overall"]
            msg += f" | dev F1 mention={res['mention']:.3f} comp={res['component']:.3f} all={score:.3f}"
        print(msg)
        history.append({"epoch": ep, "loss": run_loss / len(loader), "dev": score})
        if score is None or score > best:          # no dev set -> keep the last epoch
            best = score if score is not None else best
            torch.save(model.state_dict(), out_dir / "model.pt")
            model.encoder.config.save_pretrained(out_dir)
            tok.save_pretrained(out_dir)
            with (out_dir / "run_config.json").open("w") as f:
                json.dump({"model_short": model_short, "train_on": train_on,
                           "base_path": model_path, "max_len": max_len, "stride": stride,
                           "lr": lr, "head_lr": head_lr, "epochs": epochs,
                           "best_epoch": ep, "best_dev_overall": score,
                           "history": history}, f, indent=2)
    print(f"[encoder] saved -> {out_dir}")
    return out_dir


def load_encoder(path, device=None):
    """Load a saved run folder -> (model, tokenizer, run_config). Self-contained."""
    import torch
    from transformers import AutoTokenizer

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = json.loads((Path(path) / "run_config.json").read_text())
    tok = AutoTokenizer.from_pretrained(path, use_fast=True)
    model = build_model(str(path), pretrained=False)
    model.load_state_dict(torch.load(Path(path) / "model.pt", map_location="cpu"))
    return model.to(device).eval(), tok, cfg