from knowledge.rag.pipeline import IngestManifest


def test_manifest(tmp_path):
    m = IngestManifest(tmp_path / "manifest.json")
    assert not m.unchanged("a", "x")
    m.update("a", "x", 3)
    assert m.unchanged("a", "x")
    m2 = IngestManifest(tmp_path / "manifest.json")
    assert m2.unchanged("a", "x")
