from servelearnbench.methods import RAG, EpisodeResult
from servelearnbench.methods.rag import bm25_top_k, tokenize


def test_bm25_prefers_overlap():
    docs = [tokenize("cancel my order"), tokenize("return the blue shoes"), tokenize("exchange a lamp")]
    assert bm25_top_k("please return shoes", docs, k=2)[0] == 1
    assert bm25_top_k("unrelated words", docs, k=3) == []


def test_store_grows_only_on_update_and_ignores_timestamps():
    rag = RAG(top_k=2)
    task = type("T", (), {"instruction": "Current timestamp: 9\n\nreturn the shoes", "task_id": "t"})()
    assert rag.context(task, "serving", 0) == ""
    rag.update([EpisodeResult("a", "Current timestamp: 1\n\nreturn the shoes", 1, 1.0, "REC-A"),
                EpisodeResult("b", "Current timestamp: 2\n\ncancel an order", 2, 0.0, "REC-B")])
    text = rag.context(task, "test", 0)
    assert "REC-A" in text and "Score: 100/100." in text
    assert "stream position 1" in text
