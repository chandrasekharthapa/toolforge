import numpy as np
from fakes import DAYS_BETWEEN, draft

from toolforge.differential import compare, generate_inputs
from toolforge.embeddings import HashingEmbedder
from toolforge.knowledge import Knowledge
from toolforge.models import ToolDraft, Verification
from toolforge.registry import Registry
from toolforge.retrieval import BM25, Doc, hybrid_search
from toolforge.sandbox import Sandbox

emb = HashingEmbedder()


def test_hashing_embedder_is_deterministic_and_normalised():
    a, b = emb.embed(["convert miles to kilometers", "convert miles to kilometers"])
    assert np.allclose(a, b) and np.isclose(np.linalg.norm(a), 1.0)


def test_bm25_prefers_exact_rare_terms():
    corpus = [["sha256", "hash", "string"], ["hash", "table", "lookup"], ["string", "reverse"]]
    scores = BM25(corpus).scores(["sha256"])
    assert scores.argmax() == 0 and scores[1] == 0


def test_hybrid_search_fuses_dense_and_lexical():
    texts = ["levenshtein edit distance between two strings",
             "count vowels in a string",
             "convert temperature between celsius and fahrenheit"]
    docs = [Doc(str(i), t, emb.embed_one(t)) for i, t in enumerate(texts)]
    hits = hybrid_search("edit distance levenshtein", emb.embed_one("edit distance levenshtein"), docs, k=3)
    assert hits[0].key == "0" and hits[0].lexical_rank == 1


def test_registry_versions_and_knowledge_search(tmp_path):
    reg = Registry(str(tmp_path / "t.db"))
    kn = Knowledge(reg, emb)
    d = ToolDraft(**draft())
    v1 = kn.add_tool(d, origin_task="t", verification=Verification(tests_passed=3, tests_total=3))
    v2 = kn.add_tool(d, origin_task="t2", verification=Verification())
    assert (v1.version, v2.version) == (1, 2)
    assert [t.status for t in reg.versions("days_between")] == ["superseded", "active"]
    hits = kn.find_tools("number of days between two dates")
    assert hits[0].key == "days_between" and hits[0].payload.version == 2


def test_lessons_round_trip(tmp_path):
    kn = Knowledge(Registry(str(tmp_path / "t.db")), emb)
    kn.add_lesson("roman numerals", "forgot subtractive pairs like IV", "map IV, IX, XL, XC, CD, CM first")
    kn.add_lesson("compound interest", "used simple interest", "use p*(1+r/n)**(n*t)")
    assert kn.lessons_for("convert integer to roman numerals")[0].mistake.startswith("forgot subtractive")
    assert Knowledge(kn.registry, emb, rag=False).lessons_for("roman numerals") == []


def test_fuzz_inputs_respect_schema_and_seed_from_tests():
    d = ToolDraft(**draft())
    inputs = generate_inputs(d, 25, seed=1)
    assert len(inputs) == 25 + len(d.tests)
    assert all(set(i) == {"start", "end"} for i in inputs)
    assert inputs[0] == {"start": "2024-01-15", "end": "2024-03-01"}


def test_compare_classifies_hard_and_soft_disagreements():
    d = ToolDraft(**draft())
    off_by_one = DAYS_BETWEEN.replace("abs((b - a).days)", "abs((b - a).days) + (1 if a.year == 2024 else 0)")
    inputs = [{"start": "2024-01-01", "end": "2024-01-02"}, {"start": "2020-01-01", "end": "2020-01-02"},
              {"start": "nope", "end": "2020-01-02"}]
    report = compare(Sandbox(), d, off_by_one, inputs)
    assert len(report.hard) == 1 and report.agreements == 2 and report.soft == []


def test_test_cases_accept_common_model_shapes():
    d = ToolDraft(**draft(tests=[
        {"args": [{"start": "2024-01-15", "end": "2024-03-01"}], "expected": 46},  # gpt-oss shape
        {"input": {"start": "2024-03-01", "end": "2024-01-15"}, "output": 46},
        {"args": {"start": "2024-02-28", "end": "2024-03-01"}, "expected": 2},
    ]))
    assert [t.kwargs for t in d.tests] == [{"start": "2024-01-15", "end": "2024-03-01"},
                                           {"start": "2024-03-01", "end": "2024-01-15"},
                                           {"start": "2024-02-28", "end": "2024-03-01"}]
    assert all(t.args == [] for t in d.tests) and [t.expected for t in d.tests] == [46, 46, 2]
    assert Sandbox().run_tests(d.code, d.name, d.tests).ok


def test_single_dict_parameter_is_left_alone():
    schema = {"type": "object", "properties": {"data": {"type": "object"}}}
    d = ToolDraft(name="count_keys", description="d", parameters=schema,
                  code="def count_keys(data: dict) -> int:\n    return len(data)\n",
                  tests=[{"args": [{"data": 1}], "expected": 1}])
    assert d.tests[0].args == [{"data": 1}] and d.tests[0].kwargs == {}


def test_switching_embedder_reembeds_the_library(tmp_path):
    from toolforge.embeddings import Embedder

    class Other(Embedder):  # same dimension as the hashing embedder, different vectors
        name = "other"
        model = "x"

        def embed(self, texts, kind="passage"):
            return HashingEmbedder(dim=1024).embed([t[::-1] for t in texts])

    reg = Registry(str(tmp_path / "t.db"))
    kn = Knowledge(reg, emb)
    kn.add_tool(ToolDraft(**draft()), origin_task="t", verification=Verification())
    before = reg.tool_embeddings()[0][1]
    Knowledge(reg, Other())  # opening the library with another embedder re-embeds it
    after = reg.tool_embeddings()[0][1]
    assert reg.get_meta("embedder") == "other:x" and not np.allclose(before, after)
