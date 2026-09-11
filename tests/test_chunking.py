from core.types import DocumentBlock
from knowledge.chunking import CAEChunker


def test_heading_metadata():
    chunks = CAEChunker().chunk([
        DocumentBlock("1 热分析", "heading", 1),
        DocumentBlock("换热系数为 600 W/(m2·K)。"),
    ], "demo.md")
    assert chunks[0].metadata["section"] == "1 热分析"
    assert "600 W/(m2·K)" in chunks[0].text
