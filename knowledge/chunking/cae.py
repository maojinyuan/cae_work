"""面向 CAE 文档的结构化语义分块，不依赖 LangChain。"""
from __future__ import annotations
from core import config
from core.types import DocumentBlock, DocumentChunk

_SEPARATORS = ["\n\n", "\n", "。", "；", ";", "!", "！", "?", "？", "，", ",", "、", " "]


class CAEChunker:
    def __init__(self, chunk_size: int | None = None, overlap: int | None = None):
        self.chunk_size = chunk_size or config.CHUNK_SIZE
        self.overlap = overlap or config.CHUNK_OVERLAP

    def _split_text(self, text: str) -> list[str]:
        text = text.strip()
        if len(text) <= self.chunk_size:
            return [text] if text else []
        chunks = []
        start = 0
        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            if end < len(text):
                window = text[start:end]
                cut = -1
                for sep in _SEPARATORS:
                    pos = window.rfind(sep)
                    if pos > self.chunk_size // 2:
                        cut = pos + len(sep)
                        break
                if cut > 0:
                    end = start + cut
            piece = text[start:end].strip()
            if piece:
                chunks.append(piece)
            if end >= len(text):
                break
            start = max(end - self.overlap, start + 1)
        return chunks

    def _split_table(self, text: str) -> list[str]:
        lines = text.split("\n")
        if len(lines) <= 2:
            return [text]
        header, sep, body = lines[0], lines[1], lines[2:]
        chunks, buf, size = [], [], len(header) + len(sep)
        for row in body:
            if size + len(row) + 1 > self.chunk_size and buf:
                chunks.append("\n".join([header, sep, *buf]))
                buf, size = [], len(header) + len(sep)
            buf.append(row); size += len(row) + 1
        if buf:
            chunks.append("\n".join([header, sep, *buf]))
        return chunks or [text]

    def chunk(self, blocks: list[DocumentBlock], source: str) -> list[DocumentChunk]:
        chunks: list[DocumentChunk] = []
        section: list[str] = []
        buf: list[str] = []
        cur_page = None

        def flush():
            nonlocal buf
            if not buf:
                return
            text = "\n\n".join(buf)
            meta = {"source": source, "page": cur_page, "section": " / ".join(section), "kind": "text"}
            chunks.extend(DocumentChunk(t, dict(meta)) for t in self._split_text(text))
            buf = []

        for block in blocks:
            if block.page is not None:
                cur_page = block.page
            if block.kind == "heading":
                flush()
                level = max(block.level, 1)
                section = section[: level - 1] + [block.text]
            elif block.kind == "table":
                flush()
                meta = {"source": source, "page": cur_page, "section": " / ".join(section), "kind": "table"}
                chunks.extend(DocumentChunk(t, dict(meta)) for t in self._split_table(block.text))
            else:
                buf.append(block.text)
                if sum(map(len, buf)) >= self.chunk_size:
                    flush()
        flush()
        return chunks
