"""轻量本地 BM25；中文优先使用 jieba 分词，无 jieba 时退化为字符/英文 token。"""
from __future__ import annotations

import math
import re
from collections import Counter
from core.types import RetrievalResult


def tokenize(text: str) -> list[str]:
    text = text.lower()
    try:
        import jieba
        return [t.strip() for t in jieba.cut(text) if t.strip() and not t.isspace()]
    except ImportError:
        latin = re.findall(r"[a-z0-9_+.-]+", text)
        chinese = [c for c in text if "\u4e00" <= c <= "\u9fff"]
        return latin + chinese


class BM25Index:
    def __init__(self, docs: list[RetrievalResult], k1: float = 1.5, b: float = 0.75):
        self.docs = docs
        self.k1, self.b = k1, b
        self.tokens = [tokenize(d.text) for d in docs]
        self.lengths = [len(t) for t in self.tokens]
        self.avgdl = sum(self.lengths) / max(1, len(self.lengths))
        self.tf = [Counter(t) for t in self.tokens]
        df = Counter()
        for toks in self.tokens:
            df.update(set(toks))
        n = max(1, len(docs))
        self.idf = {term: math.log(1 + (n - freq + 0.5) / (freq + 0.5)) for term, freq in df.items()}

    def search(self, query: str, top_k: int) -> list[RetrievalResult]:
        q = tokenize(query)
        scored = []
        for i, tf in enumerate(self.tf):
            dl = self.lengths[i] or 1
            score = 0.0
            for term in q:
                f = tf.get(term, 0)
                if not f:
                    continue
                denom = f + self.k1 * (1 - self.b + self.b * dl / max(self.avgdl, 1e-9))
                score += self.idf.get(term, 0.0) * (f * (self.k1 + 1)) / denom
            if score > 0:
                d = self.docs[i]
                scored.append(RetrievalResult(d.text, d.source, d.page, d.section, d.kind, score))
        scored.sort(key=lambda x: x.score, reverse=True)
        return scored[:top_k]
