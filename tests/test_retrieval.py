from core.types import RetrievalResult
from knowledge.retrieval.bm25 import BM25Index
from knowledge.retrieval.retriever import reciprocal_rank_fusion


def hit(text, score, source="a.txt"):
    return RetrievalResult(text=text, source=source, page=1, section="", kind="text", score=score)


def test_bm25_prefers_matching_doc():
    docs = [hit("发动机缸盖 热应力 温度边界条件", 0), hit("排气系统 声学 NVH", 0)]
    out = BM25Index(docs).search("缸盖热应力边界条件", 2)
    assert out
    assert "缸盖" in out[0].text


def test_rrf_merges_duplicate_results():
    a = hit("same", 0.9)
    b = hit("other", 0.8, source="b.txt")
    out = reciprocal_rank_fusion([a, b], [a])
    assert len(out) == 2
    assert out[0].text == "same"
