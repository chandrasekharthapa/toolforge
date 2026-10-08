"""Distil Toolforge's LLM reuse-judge into a small cross-encoder reranker.

Input: the dataset built by ``python -m evals.distill_data`` (evals/distill/).
Model: a cross-encoder (default ``cross-encoder/ms-marco-MiniLM-L6-v2``, ~23M parameters, already trained for
       query/passage relevance) fine-tuned to score (need, tool card) pairs.

Decision rule, identical for the zero-shot baseline and the fine-tuned model:
    score every candidate tool; reuse the best one if sigmoid(score) >= tau, otherwise build a new tool.
tau is chosen on the VALIDATION split (tools never seen in training) to best match the teacher.

Reported on the TEST split (the hand-labelled 65 needs over 32 tools absent from training):
  * decision accuracy against human gold labels, for the teacher, the zero-shot model and the student
  * agreement with the teacher, per-decision latency (GPU and CPU), parameter count

    python evals/train_reranker.py                         # GPU if available
    python evals/train_reranker.py --epochs 4 --lr 3e-5    # hyper-parameters are flags
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

ROOT = Path(__file__).resolve().parents[1]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# --------------------------------------------------------------------------------------- scoring

@torch.no_grad()
def score_pairs(model, tokenizer, queries: list[str], cards: list[str], device: str, max_len: int,
                batch: int = 64) -> np.ndarray:
    model.eval()
    out = []
    for i in range(0, len(queries), batch):
        enc = tokenizer(queries[i:i + batch], cards[i:i + batch], padding=True, truncation=True,
                        max_length=max_len, return_tensors="pt").to(device)
        out.append(model(**enc).logits.squeeze(-1).float().cpu())
    return torch.sigmoid(torch.cat(out)).numpy() if out else np.zeros(0)


def decision_scores(model, tokenizer, decisions: list[dict[str, Any]], device: str, max_len: int) -> list[np.ndarray]:
    """Per decision: one probability per candidate tool."""
    q = [d["query"] for d in decisions for _ in d["candidate_texts"]]
    c = [t for d in decisions for t in d["candidate_texts"]]
    flat = score_pairs(model, tokenizer, q, c, device, max_len)
    out, k = [], 0
    for d in decisions:
        n = len(d["candidate_texts"])
        out.append(flat[k:k + n])
        k += n
    return out


def decide(decision: dict[str, Any], probs: np.ndarray, tau: float) -> str | None:
    if len(probs) == 0:
        return None
    best = int(np.argmax(probs))
    return decision["candidates"][best]["name"] if probs[best] >= tau else None


def accuracy(decisions: list[dict[str, Any]], scores: list[np.ndarray], tau: float, key: str) -> float:
    return statistics.fmean(decide(d, s, tau) == d[key] for d, s in zip(decisions, scores)) if decisions else 0.0


def choose_tau(decisions: list[dict[str, Any]], scores: list[np.ndarray], key: str = "teacher") -> float:
    """Threshold maximising agreement on validation; ties broken toward the middle of the best interval."""
    tops = sorted({float(s.max()) for s in scores if len(s)})
    grid = [0.0] + [(a + b) / 2 for a, b in zip(tops, tops[1:])] + [1.0 + 1e-6]
    accs = [accuracy(decisions, scores, t, key) for t in grid]
    best = max(accs)
    ties = [t for t, a in zip(grid, accs) if a == best]
    return ties[len(ties) // 2]


@torch.no_grad()
def latency_ms(model, tokenizer, decisions: list[dict[str, Any]], device: str, max_len: int, n: int = 40) -> float:
    """Median wall time of one reuse decision (score all candidates of one need) on ``device``."""
    model.eval().to(device)
    times = []
    for d in (decisions * math.ceil(n / max(1, len(decisions))))[:n]:
        t0 = time.perf_counter()
        enc = tokenizer([d["query"]] * len(d["candidate_texts"]), d["candidate_texts"], padding=True,
                        truncation=True, max_length=max_len, return_tensors="pt").to(device)
        model(**enc)
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)
    return float(statistics.median(times[3:] or times))  # drop warm-up


# --------------------------------------------------------------------------------------- training

def train(model, tokenizer, pairs: list[dict[str, Any]], val_dec: list[dict[str, Any]], args, device: str,
          log=print) -> dict[str, Any]:
    pos = sum(p["label"] for p in pairs)
    neg = len(pairs) - pos
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(neg / max(pos, 1), device=device))
    loader = DataLoader(pairs, batch_size=args.batch, shuffle=True, collate_fn=lambda b: b,
                        generator=torch.Generator().manual_seed(args.seed))
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total = len(loader) * args.epochs
    sched = get_linear_schedule_with_warmup(opt, int(total * 0.1), total)
    best = {"val_acc": -1.0, "epoch": 0, "state": None}
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for batch in loader:
            enc = tokenizer([p["query"] for p in batch], [p["card"] for p in batch], padding=True, truncation=True,
                            max_length=args.max_len, return_tensors="pt").to(device)
            labels = torch.tensor([float(p["label"]) for p in batch], device=device)
            loss = loss_fn(model(**enc).logits.squeeze(-1), labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad()
            losses.append(loss.item())
        scores = decision_scores(model, tokenizer, val_dec, device, args.max_len)
        tau = choose_tau(val_dec, scores)
        val_acc = accuracy(val_dec, scores, tau, "teacher")
        history.append({"epoch": epoch, "train_loss": round(statistics.fmean(losses), 4), "val_agreement": val_acc})
        log(f"epoch {epoch}: train loss {statistics.fmean(losses):.4f} · val agreement with teacher {val_acc:.1%}")
        if val_acc > best["val_acc"]:
            best = {"val_acc": val_acc, "epoch": epoch,
                    "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}}
    model.load_state_dict(best["state"])
    return {"history": history, "best_epoch": best["epoch"], "pos_weight": round(neg / max(pos, 1), 3)}


def load_model(name: str, device: str, tiny: bool = False, workdir: Path | None = None, data: Path | None = None):
    if tiny:  # offline smoke test: a randomly initialised miniature BERT, no download
        from transformers import BertConfig, BertForSequenceClassification, BertTokenizerFast

        (workdir or ROOT).mkdir(parents=True, exist_ok=True)
        vocab = (workdir or ROOT) / ".tiny_vocab.txt"
        source = data / "pairs_train.jsonl" if data and (data / "pairs_train.jsonl").exists() else (
            ROOT / "evals" / "retrieval_set.json")
        words = sorted({w for w in source.read_text(encoding="utf-8").lower().split() if w.isalpha()})
        vocab.write_text("\n".join(["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", *words]))
        tok = BertTokenizerFast(vocab_file=str(vocab))
        cfg = BertConfig(vocab_size=tok.vocab_size, hidden_size=32, num_hidden_layers=2, num_attention_heads=2,
                         intermediate_size=64, num_labels=1)
        return BertForSequenceClassification(cfg).to(device), tok
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForSequenceClassification.from_pretrained(name, num_labels=1)
    return model.to(device), tok


def evaluate(model, tok, val_dec, test_dec, device, max_len) -> dict[str, Any]:
    val_s = decision_scores(model, tok, val_dec, device, max_len)
    tau = choose_tau(val_dec, val_s)
    test_s = decision_scores(model, tok, test_dec, device, max_len)
    pos = [d for d in test_dec if d["gold"]]
    neg = [d for d in test_dec if not d["gold"]]
    picks = [decide(d, s, tau) for d, s in zip(test_dec, test_s)]
    by = dict(zip(map(id, test_dec), picks))
    return {"tau": round(tau, 4),
            "val_agreement_with_teacher": round(accuracy(val_dec, val_s, tau, "teacher"), 3),
            "test_accuracy_vs_gold": round(accuracy(test_dec, test_s, tau, "gold"), 3),
            "test_agreement_with_teacher": round(accuracy(test_dec, test_s, tau, "teacher"), 3),
            "test_positives_correct": sum(by[id(d)] == d["gold"] for d in pos), "test_positives": len(pos),
            "test_negatives_correct": sum(by[id(d)] is None for d in neg), "test_negatives": len(neg)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=str(ROOT / "evals" / "distill"))
    ap.add_argument("--model", default="cross-encoder/ms-marco-MiniLM-L6-v2")
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--max-len", type=int, default=192)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None, help="where to save the model (default: <data>/reranker)")
    ap.add_argument("--results", default=str(ROOT / "evals" / "results"), help="where reranker.md/json go")
    ap.add_argument("--tiny", action="store_true", help="offline smoke test with a random miniature model")
    args = ap.parse_args()

    seed_everything(args.seed)
    data = Path(args.data)
    out = Path(args.out) if args.out else data / "reranker"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pairs = read_jsonl(data / "pairs_train.jsonl")
    val_dec = read_jsonl(data / "decisions_val.jsonl")
    test_dec = read_jsonl(data / "decisions_test.jsonl")
    stats = json.loads((data / "stats.json").read_text(encoding="utf-8"))
    print(f"device {device} · {len(pairs)} training pairs ({sum(p['label'] for p in pairs)} positive) · "
          f"{len(val_dec)} validation decisions · {len(test_dec)} test decisions")

    model, tok = load_model(args.model, device, args.tiny, workdir=out, data=Path(args.data))
    params = sum(p.numel() for p in model.parameters())
    zero_shot = evaluate(model, tok, val_dec, test_dec, device, args.max_len)
    print(f"zero-shot: test accuracy {zero_shot['test_accuracy_vs_gold']:.1%}")

    t0 = time.perf_counter()
    training = train(model, tok, pairs, val_dec, args, device)
    train_s = time.perf_counter() - t0
    student = evaluate(model, tok, val_dec, test_dec, device, args.max_len)
    print(f"fine-tuned: test accuracy {student['test_accuracy_vs_gold']:.1%}")

    lat = {"device": device, "ms_per_decision": round(latency_ms(model, tok, test_dec, device, args.max_len), 2)}
    if device != "cpu":
        lat["cpu_ms_per_decision"] = round(latency_ms(model, tok, test_dec, "cpu", args.max_len), 2)
        model.to(device)
    else:
        lat["cpu_ms_per_decision"] = lat["ms_per_decision"]

    result = {"base_model": "tiny-random" if args.tiny else args.model, "parameters": params,
              "train_seconds": round(train_s, 1), "hyperparameters": {k: getattr(args, k) for k in
                                                                     ("epochs", "lr", "batch", "max_len", "seed")},
              "teacher": {"test_accuracy_vs_gold": stats.get("teacher_test_accuracy"),
                          "mean_latency_s": stats.get("teacher_mean_latency_s"),
                          "mean_tokens": stats.get("teacher_mean_tokens")},
              "zero_shot": zero_shot, "student": student, "latency": lat, "training": training,
              "data": stats.get("splits")}
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out)
    tok.save_pretrained(out)
    (out / "toolforge_reranker.json").write_text(json.dumps({"threshold": student["tau"], **result}, indent=1),
                                                 encoding="utf-8")
    results_dir = Path(args.results)
    results_dir.mkdir(exist_ok=True)
    (results_dir / "reranker.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
    md = render(result)
    (results_dir / "reranker.md").write_text(md, encoding="utf-8")
    print("\n" + md)
    return 0


def render(r: dict[str, Any]) -> str:
    t, z, s, lat = r["teacher"], r["zero_shot"], r["student"], r["latency"]
    teacher_ms = f"{t['mean_latency_s'] * 1000:,.0f} ms" if t.get("mean_latency_s") else "–"
    return "\n".join([
        f"## Distilling the reuse judge: `{r['base_model']}` ({r['parameters'] / 1e6:.1f}M parameters)", "",
        "Test: 65 hand-labelled needs over 32 tools that never appear in training; same candidate lists for all.",
        "", "| judge | decision accuracy (human labels) | agrees with teacher | cost per decision | latency |",
        "|---|---|---|---|---|",
        f"| teacher LLM judge | {t['test_accuracy_vs_gold']:.0%} | — | {t.get('mean_tokens') or 0:,.0f} tokens | "
        f"{teacher_ms} |",
        f"| cross-encoder, zero-shot | {z['test_accuracy_vs_gold']:.0%} | {z['test_agreement_with_teacher']:.0%} | "
        f"0 tokens | {lat['ms_per_decision']} ms ({lat['device']}) |",
        f"| **cross-encoder, fine-tuned** | **{s['test_accuracy_vs_gold']:.0%}** | "
        f"{s['test_agreement_with_teacher']:.0%} | **0 tokens** | **{lat['ms_per_decision']} ms** ({lat['device']})"
        + ("" if lat["device"] == "cpu" else f", {lat['cpu_ms_per_decision']} ms CPU") + " |", "",
        f"Fine-tuned: {s['test_positives_correct']}/{s['test_positives']} needs with a tool reused the right one, "
        f"{s['test_negatives_correct']}/{s['test_negatives']} needs without one correctly built a new tool. "
        f"Threshold {s['tau']} chosen on validation tools (unseen in training). Trained {r['train_seconds']} s, "
        f"best epoch {r['training']['best_epoch']}.", ""])


if __name__ == "__main__":
    raise SystemExit(main())
