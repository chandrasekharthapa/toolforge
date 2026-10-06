import json
import re

from evals.retrieval import gate_decision, load_set, run
from toolforge.config import Settings
from toolforge.embeddings import Embedder, HashingEmbedder
from toolforge.llm import ScriptedLLM


def test_offline_retrieval_benchmark_shape_and_floor():
    result = run(HashingEmbedder())
    r = result["ranking"]
    assert {"bm25", "dense", "hybrid"} <= set(r) and {"hybrid@1.0", "hybrid@0.1"} <= set(r)
    assert r["hybrid"] == r["hybrid@1.0"]  # the shipped hashing weight is 1.0
    assert r["hybrid"]["hit@1"] >= 0.85  # regression floor for the offline embedder on this set
    assert r["hybrid"]["mrr"] >= max(r["bm25"]["mrr"], r["dense"]["mrr"]) - 1e-9  # fusion never hurts here
    g = result["gate"]
    assert g["negatives"]["auto_reused_wrongly"] == 0  # auto-reuse must never pick a tool for a no-tool need
    assert g["positives"]["total"] == 50 and g["negatives"]["total"] == 15


def test_dataset_is_well_formed():
    tools, queries = load_set()
    names = {t.name for t in tools}
    assert len(names) == len(tools) == 32
    assert all(q["gold"] in names for q in queries if q["gold"])
    assert sum(1 for q in queries if not q["gold"]) == 15


def test_gate_decision_mirrors_the_agent():
    row = {"top_k": [("a", 0.50), ("b", 0.91), ("c", 0.10)]}
    assert gate_decision(row, reuse_t=0.9, consider_t=0.2) == ("auto_reuse", "b")  # highest cosine, not rank 1
    assert gate_decision(row, reuse_t=0.95, consider_t=0.2) == ("judge", None)
    assert gate_decision(row, reuse_t=0.95, consider_t=0.95) == ("create", None)


def test_end_to_end_judge_metrics_use_the_real_match_step():
    _, queries = load_set()
    gold = {q["description"]: q["gold"] for q in queries}

    def judge(system, prompt):
        task = re.search(r"Task: (.*)\n", prompt).group(1)
        shown = [c["name"] for c in json.loads(prompt.split("Candidate tools:\n", 1)[1])]
        want = gold[task]
        return {"choice": want if want in shown else None, "reason": "scripted oracle"}

    settings = Settings(provider="scripted", embedder="hashing")
    result = run(HashingEmbedder(), llm=ScriptedLLM(judge), settings=settings, log=lambda _: None)
    j = result["judge"]
    assert j["negatives"]["reused_wrongly"] == 0  # a perfect judge never reuses for a no-tool need
    assert j["positives"]["reused_wrong_tool"] == 0
    # with a perfect judge, the only losses are needs whose correct tool was never shown to it
    gate = result["gate"]["positives"]
    shown = gate["judge_shown_correct_tool"] + gate["auto_reused_correctly"]
    assert j["positives"]["reused_correct_tool"] == shown
    assert "End to end, with the LLM judge" in result["markdown"]


class FakeAsymmetric(Embedder):
    name = "fake"
    model = "m1"

    def __init__(self):
        self.kinds = []

    def embed(self, texts, kind="passage"):
        self.kinds.append(kind)
        return HashingEmbedder().embed(texts)


def test_queries_and_passages_are_embedded_as_such():
    emb = FakeAsymmetric()
    run(emb)
    assert "query" in emb.kinds and "passage" in emb.kinds


def test_probe_reports_which_embedding_models_work():
    import httpx

    from toolforge.embeddings import EmbeddingError, NvidiaEmbedder

    def handler(req):
        if req.url.path.endswith("/models"):
            return httpx.Response(200, json={"data": [{"id": "nvidia/old-embedqa"}, {"id": "nvidia/new-embed"},
                                                      {"id": "nvidia/chat-model"}]})
        body = json.loads(req.content)
        if body["model"] == "nvidia/old-embedqa":
            return httpx.Response(404, json={"detail": "Function not found"})
        assert body["input_type"] == "query"  # asymmetric model gets the query/passage flag
        return httpx.Response(200, json={"data": [{"index": 0, "embedding": [0.6, 0.8, 0.0]}]})

    emb = NvidiaEmbedder(model="nvidia/old-embedqa")
    emb.http = httpx.Client(transport=httpx.MockTransport(handler))
    assert emb.probe() == [("nvidia/new-embed", True, "dim 3"),
                           ("nvidia/old-embedqa", False,
                            "nvidia embeddings failed for model 'nvidia/old-embedqa' (HTTP 404): "
                            '{"detail":"Function not found"}')]
    assert emb.model == "nvidia/old-embedqa"  # probing leaves the configured model unchanged
    try:
        emb.embed(["x"])
    except EmbeddingError as e:
        assert "--probe" in str(e)


def test_lexical_weight_reaches_the_agent_and_the_eval():
    from toolforge.knowledge import Knowledge
    from toolforge.registry import Registry

    kn = Knowledge(Registry(":memory:"), HashingEmbedder(), lexical_weight=0.25)
    assert kn.lexical_weight == 0.25
    assert Knowledge(Registry(":memory:"), HashingEmbedder()).lexical_weight == HashingEmbedder.lexical_weight
    low = run(HashingEmbedder(), lexical_weight=0.1)["ranking"]
    assert low["hybrid"] == low["hybrid@0.1"]
