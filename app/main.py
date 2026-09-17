"""CAE AI Platform FastAPI 服务。"""
from __future__ import annotations
import shutil
from threading import Lock
from pathlib import Path
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from core import config
from knowledge.rag import get_pipeline
from knowledge.vectorstore import get_store

# ponytail: serialize generation for a single local GPU; use batching if throughput matters.
_query_lock = Lock()

APP_DIR = Path(__file__).resolve().parent
STATIC_DIR = APP_DIR / "static"
app = FastAPI(title="CAE AI Platform", version="0.2.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


class QueryRequest(BaseModel):
    question: str
    top_k: int | None = None


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/health")
def health():
    return {"status": "ok", "stats": get_store().stats()}


@app.post("/api/query")
def query(request: QueryRequest):
    question = request.question.strip()
    if not question:
        raise HTTPException(400, "question 不能为空")
    with _query_lock:
        return get_pipeline().ask(question, request.top_k)


@app.post("/api/search")
def search(request: QueryRequest):
    question = request.question.strip()
    if not question:
        raise HTTPException(400, "question 不能为空")
    hits = get_pipeline().search(question, request.top_k)
    return {"retrieved": len(hits), "results": [h.__dict__ for h in hits]}


@app.post("/api/ingest")
async def ingest(files: list[UploadFile] = File(...)):
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    imported, errors = [], []
    for upload in files:
        name = Path(upload.filename or "").name
        if not name or name != upload.filename:
            errors.append({"file": upload.filename, "error": "非法文件名"}); continue
        dest = config.DATA_DIR / name
        with open(dest, "wb") as output:
            shutil.copyfileobj(upload.file, output)
        try:
            result = get_pipeline().ingest_file(dest)
            (imported if result["status"] == "ok" else errors).append(result)
        except Exception as exc:
            errors.append({"file": name, "error": str(exc)})
    return {"imported": imported, "errors": errors, "total_chunks": sum(x.get("chunks", 0) for x in imported)}


app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
