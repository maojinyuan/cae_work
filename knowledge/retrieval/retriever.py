"""Dense / Hybrid retrieval + optional reranking。"""
from __future__ import annotations
from core import config
from core.types import RetrievalResult
from knowledge.retrieval.bm25 import BM25Index


def _key(hit: RetrievalResult) -> tuple:
    return (hit.source, hit.page, hit.section, hit.text)


def reciprocal_rank_fusion(dense, lexical):
    scores = {}
    objects = {}
    for weight, items in ((config.HYBRID_DENSE_WEIGHT, dense), (config.HYBRID_LEXICAL_WEIGHT, lexical)):
        for rank, hit in enumerate(items, 1):
            k = _key(hit)
            objects[k] = hit
            scores[k] = scores.get(k, 0.0) + weight / (config.RRF_K + rank)
    ordered = sorted(scores, key=scores.get, reverse=True)
    out = []
    for k in ordered:
        h = objects[k]
        out.append(RetrievalResult(h.text, h.source, h.page, h.section, h.kind, round(scores[k], 8)))
    return out


class Retriever:
    def __init__(self, embedder, store):
        self.embedder = embedder
        self.store = store
        self._bm25 = None
        self._bm25_count = -1
        self._reranker = None

    def _lexical(self, question: str, k: int):
        count = self.store.stats()["count"]
        if self._bm25 is None or count != self._bm25_count:
            self._bm25 = BM25Index(self.store.all_records())
            self._bm25_count = count
        return self._bm25.search(question, k)

    def _get_reranker(self):
        if self._reranker is None:
            from knowledge.reranking import LocalReranker
            self._reranker = LocalReranker()
        return self._reranker

    def retrieve(self, question: str, top_k: int | None = None, threshold: float | None = None):
        final_k = top_k or config.TOP_K
        candidate_k = max(final_k, config.CANDIDATE_K)
        embedding = self.embedder.embed_one(question)

        if config.RETRIEVAL_MODE == "hybrid":
            dense = self.store.query(embedding, candidate_k, threshold=None)
            lexical = self._lexical(question, candidate_k)
            hits = reciprocal_rank_fusion(dense, lexical)[:candidate_k]
        else:
            th = config.SIM_THRESHOLD if threshold is None else threshold
            hits = self.store.query(embedding, candidate_k if config.RERANK_ENABLED else final_k, th)

        if config.RERANK_ENABLED:
            return self._get_reranker().rerank(question, hits, final_k)
        return hits[:final_k]
