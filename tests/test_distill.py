import json
import re

import evals.distill_data as D
from evals.retrieval import load_set
from toolforge.embeddings import HashingEmbedder
from toolforge.llm import ScriptedLLM

NEW_TOOLS = [
    {"name": "circle_area", "description": "Area of a circle from its radius.",
     "parameters": {"type": "object", "properties": {"radius": {"type": "number"}}}},
    {"name": "hex_to_rgb", "description": "Convert a hex colour string such as #ff8800 to an RGB triple.",
     "parameters": {"type": "object", "properties": {"hex": {"type": "string"}}}},
    {"name": "reverse_words", "description": "Reverse the order of the words in a sentence.",
     "parameters": {"type": "object", "properties": {"text": {"type": "string"}}}},
    {"name": "is_leap_year", "description": "Whether a year is a leap year in the Gregorian calendar.",
     "parameters": {"type": "object", "properties": {"year": {"type": "integer"}}}},
    {"name": "day_of_week_name", "description": "English weekday name for a date.",  # leaks: eval has day_of_week
     "parameters": {"type": "object", "properties": {"date": {"type": "string"}}}},
]


def brain(system, prompt):
    if system == D.TOOLS_SYSTEM:
        return {"tools": NEW_TOOLS if "text processing" in prompt else []}
    if system == D.NEEDS_SYSTEM:
        name = re.search(r"Tool: (\w+):", prompt).group(1)
        return {"satisfied": [{"name_hint": f"{name}_{i}", "description": f"please compute {name} variant {i}"}
                              for i in range(3)],
                "near_miss": [{"name_hint": f"not_{name}", "description": f"something unlike {name}"}]}
    # teacher judge: picks the candidate the need was written for, if it was retrieved
    need = re.search(r"Need: (\w+)", prompt).group(1)
    shown = [c["name"] for c in json.loads(prompt.split("Candidate tools:\n", 1)[1])]
    pick = next((s for s in shown if need.startswith(s + "_") or need == s), None)
    return {"choice": pick, "reason": "scripted"}


def test_distillation_pipeline_end_to_end(tmp_path, monkeypatch):
    monkeypatch.setattr(D, "OUT", tmp_path)
    llm, emb = ScriptedLLM(brain), HashingEmbedder()
    eval_tools, eval_queries = load_set()

    tools = D.generate_tools(llm, emb, eval_tools, log=lambda _: None)
    names = [t["name"] for t in tools]
    assert "day_of_week_name" not in names and "circle_area" in names  # leakage guard
    dropped = json.loads((tmp_path / "tools.json").read_text())["dropped"]
    assert any(d["name"] == "day_of_week_name" for d in dropped)

    needs = D.generate_needs(llm, tools, log=lambda _: None)
    assert len(needs) == len(tools) * 4
    assert D.generate_needs(llm, tools, log=lambda _: None) == needs  # resumable, no new calls
    train = D.label(llm, emb, [D._draft(t) for t in tools], needs, "labelled_train.jsonl", 1.0, log=lambda _: None)
    assert all(len(r["candidates"]) == min(D.TOP_K, len(tools)) for r in train)
    assert all(r["teacher"] is None for r in train if r["kind"] == "near_miss")

    test_needs = [{"source_tool": "", "kind": "test", **q} for q in eval_queries[:6]]
    test = D.label(llm, emb, eval_tools, test_needs, "labelled_test.jsonl", 1.0, log=lambda _: None)
    stats = D.export(train, test)

    tr = [json.loads(x) for x in (tmp_path / "pairs_train.jsonl").read_text().splitlines()]
    va = [json.loads(x) for x in (tmp_path / "pairs_val.jsonl").read_text().splitlines()]
    assert {p["group"] for p in tr}.isdisjoint({p["group"] for p in va})  # split by tool: no tool in both
    assert sum(p["label"] for p in tr + va) == sum(1 for r in train if r["teacher"])  # one positive per pick
    assert stats["test_needs"] == 6 and 0.0 <= stats["teacher_test_accuracy"] <= 1.0
    assert stats["teacher_calls"] == len(train) + 6 and stats["teacher_mean_tokens"] > 0
    dec = [json.loads(x) for x in (tmp_path / "decisions_test.jsonl").read_text().splitlines()]
    assert all({"query", "candidate_texts", "gold", "teacher"} <= set(d) for d in dec)


def test_colab_bundle_has_posix_paths(tmp_path, monkeypatch):
    import zipfile

    monkeypatch.setattr(D, "OUT", tmp_path)
    (tmp_path / "pairs_train.jsonl").write_text("{}\n")
    (tmp_path / "stats.json").write_text("{}")
    (tmp_path / "notes.txt").write_text("not shipped")
    names = zipfile.ZipFile(D.colab_bundle(tmp_path / "b.zip")).namelist()
    assert "evals/train_reranker.py" in names and "evals/distill/pairs_train.jsonl" in names
    assert "evals/distill/stats.json" in names and not any("\\" in n or n.endswith(".txt") for n in names)
