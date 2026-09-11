from knowledge.cache import EmbeddingCache


def test_embedding_cache_roundtrip(tmp_path):
    cache = EmbeddingCache(tmp_path / "emb.sqlite3", "unit")
    cache.put_many(["a", "b"], [[1.0, 2.0], [3.0, 4.0]])
    got = cache.get_many(["b", "missing", "a"])
    assert got[0] == [3.0, 4.0]
    assert 1 not in got
    assert got[2] == [1.0, 2.0]
