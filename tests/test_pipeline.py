from pathlib import Path
from core.types import RetrievalResult
from knowledge.rag.pipeline import RAGPipeline


class FakeEmbedder:
    def embed(self, texts): return [[float(len(t)), 1.0] for t in texts]
    def embed_one(self, text): return [float(len(text)), 1.0]


class FakeStore:
    def __init__(self): self.rows=[]
    def reset(self): self.rows=[]
    def delete_source(self, source): self.rows=[r for r in self.rows if r.source != source]
    def upsert(self, chunks, embeddings, source):
        self.delete_source(source)
        self.rows.extend(RetrievalResult(c.text, source, c.metadata.get('page'), c.metadata.get('section',''), c.metadata.get('kind','text'), 0.9) for c in chunks)
        return len(chunks)
    def query(self, embedding, top_k, threshold=None): return self.rows[:top_k]
    def all_records(self): return self.rows
    def stats(self): return {'count':len(self.rows)}


class FakeLLM:
    def generate(self, messages, max_tokens=None):
        return 'fake answer'


def test_fake_end_to_end(tmp_path, monkeypatch):
    import core.config as cfg
    manifest = tmp_path / 'manifest.json'
    monkeypatch.setattr(cfg, 'INGEST_MANIFEST', manifest)
    monkeypatch.setattr(cfg, 'INGEST_SKIP_UNCHANGED', True)
    monkeypatch.setattr(cfg, 'RETRIEVAL_MODE', 'dense')
    monkeypatch.setattr(cfg, 'RERANK_ENABLED', False)
    doc = tmp_path / 'demo.txt'
    doc.write_text('缸盖热应力分析的温度边界条件。', encoding='utf-8')
    rag = RAGPipeline(embedder=FakeEmbedder(), store=FakeStore(), llm=FakeLLM())
    result = rag.ingest_file(doc)
    assert result['status'] == 'ok'
    second = rag.ingest_file(doc)
    assert second['status'] == 'unchanged'
    answer = rag.ask('温度边界条件是什么')
    assert answer['answer'] == 'fake answer'
    assert answer['retrieved'] >= 1
