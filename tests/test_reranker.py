import json
import sys

import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")

from fakes import DAYS_BETWEEN, Brain, draft  # noqa: E402
from test_distill import brain as distill_brain  # noqa: E402

import evals.distill_data as D  # noqa: E402
from evals.retrieval import load_set  # noqa: E402
from toolforge.embeddings import HashingEmbedder  # noqa: E402
from toolforge.llm import ScriptedLLM  # noqa: E402


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    out = tmp_path_factory.mktemp("distill")
    saved_out, D.OUT = D.OUT, out
    llm, emb = ScriptedLLM(distill_brain), HashingEmbedder()
    tools_eval, queries = load_set()
    tools = D.generate_tools(llm, emb, tools_eval, log=lambda _: None)
    needs = D.generate_needs(llm, tools, log=lambda _: None)
    train = D.label(llm, emb, [D._draft(t) for t in tools], needs, "labelled_train.jsonl", 1.0, log=lambda _: None)
    test = D.label(llm, emb, tools_eval, [{"source_tool": "", "kind": "test", **q} for q in queries[:8]],
                   "labelled_test.jsonl", 1.0, log=lambda _: None)
    D.export(train, test)
    import evals.train_reranker as T

    argv = sys.argv
    sys.argv = ["train_reranker", "--tiny", "--data", str(out), "--epochs", "1", "--batch", "8",
                "--out", str(out / "reranker"), "--results", str(out / "results")]
    try:
        assert T.main() == 0
    finally:
        sys.argv = argv
        D.OUT = saved_out
    return out / "reranker"


def test_training_script_writes_model_and_report(trained):
    meta = json.loads((trained / "toolforge_reranker.json").read_text())
    assert {"threshold", "zero_shot", "student", "latency", "parameters"} <= set(meta)
    assert 0.0 <= meta["student"]["test_accuracy_vs_gold"] <= 1.0
    assert (trained.parent / "results" / "reranker.md").exists()


def test_reranker_judge_respects_its_threshold(trained):
    from toolforge.reranker import RerankerJudge

    class Card:
        def __init__(self, name):
            self.name = name

        def search_text(self):
            return f"{self.name.replace('_', ' ')}. does {self.name}"

    cands = [Card("circle_area"), Card("hex_to_rgb")]
    pick, _ = RerankerJudge(str(trained), threshold=0.0).choose("area of a circle", cands)
    assert pick in {"circle_area", "hex_to_rgb"}
    assert RerankerJudge(str(trained), threshold=1.01).choose("area of a circle", cands)[0] is None


def test_agent_uses_reranker_instead_of_llm_judge(trained, make_forge):
    from toolforge.reranker import RerankerJudge

    b = Brain().plan("How many days", ("days_between", "Number of days between two ISO dates."))
    b.references["days_between"] = DAYS_BETWEEN
    b.will_write("days_between", draft())
    forge = make_forge(b)
    forge.run("How many days between 2024-01-15 and 2024-03-01?")
    forge.forge.reranker = RerankerJudge(str(trained), threshold=0.0)  # accept its best candidate
    result = forge.run("How many days between 2023-12-25 and 2024-02-14?")
    assert "judge" not in b.roles  # the LLM judge was never called
    assert result.reused == ["days_between"]
    match = next(e for e in result.trace if e["node"] == "match")
    assert match["reason"].startswith("reranker")
