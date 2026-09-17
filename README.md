# CAE AI Platform

面向企业 CAE 仿真的本地优先 AI/RAG 基础平台。当前版本重点是 **离线知识库 + RAG**：默认使用本地 LLM、本地 Embedding、本地 ChromaDB；在有网络或公司内部模型服务时，可以只改 `.env` 切换到 OpenAI-compatible API。

> 推荐先把这一层跑稳定，再向强度/NVH/CFD Agent、ML/ROM、求解器调用扩展。

---

## 1. 这版重构做了什么

### 默认主路线

```text
CAE documents
   ↓
Loader (PDF/DOCX/PPTX/TXT/MD)
   ↓
CAE-aware Chunker
   ↓
Local Embedding (SentenceTransformers)
   ↓
Embedding Cache (SQLite)
   ↓
ChromaDB
   ↓
Dense / Hybrid Retrieval
   ↓
Optional Local Reranker
   ↓
Local LLM (Transformers)
   ↓
Answer + Sources
```

### Provider 切换

```text
LLM
├── local  -> Hugging Face Transformers 本地目录
└── api    -> OpenAI-compatible API

Embedding
├── local  -> SentenceTransformers 本地目录
└── api    -> OpenAI-compatible Embedding API
```

`fastembed` 已从主架构和默认依赖中移除。它不是不能用，而是当前项目更适合统一围绕 `SentenceTransformers / Transformers` 建立本地模型运行层，后续再按硬件需要增加 ONNX/OpenVINO/TensorRT 等专门后端，而不是同时维护多套语义不完全一致的 embedding provider。

---

## 2. RAG 方面新增的优化

### 已实现

1. **Embedding 缓存**：同一文本不重复编码，SQLite 持久化。
2. **增量入库**：通过文件 SHA256 manifest 跳过未修改文件。
3. **批量写 Chroma**：减少大量 chunk 一次写入的压力。
4. **Dense Retrieval**：默认模式，最轻、最快。
5. **Hybrid Retrieval**：可选 `Dense + BM25 + RRF`。
6. **Local Reranker**：可选 CrossEncoder 二阶段重排。
7. **Candidate-K / Final-K 分离**：先召回更多候选，再 rerank 到最终 top-k。
8. **本地强制离线**：设置 `HF_HUB_OFFLINE=1`、`TRANSFORMERS_OFFLINE=1`，模型加载使用 `local_files_only=True`。

### 推荐的质量升级顺序

```text
Dense only
   ↓
Dense + larger candidate_k
   ↓
Hybrid (Dense + BM25 + RRF)
   ↓
Hybrid + Reranker
   ↓
Query rewrite / multi-query
   ↓
RAG evaluation + threshold tuning
```

不要一开始把所有开关都打开。企业 CAE 文档通常有大量型号、材料牌号、零部件名称、边界条件和数值参数，Hybrid 对这类关键词/编号检索通常很有价值；Reranker 则在候选较相近时提升排序质量，但会增加一次模型推理。

---

## 3. 目录结构

```text
CAE_AI_Platform/
├── app/                         # FastAPI / Web
├── core/                        # 配置、类型、异常
├── knowledge/
│   ├── loaders/                 # 文档解析
│   ├── chunking/                # CAE 分块
│   ├── embedding/               # local / api embedding
│   ├── cache/                   # embedding cache
│   ├── vectorstore/             # ChromaDB
│   ├── retrieval/               # dense / BM25 / hybrid
│   ├── reranking/               # optional reranker
│   └── rag/                     # prompt + pipeline
├── models/
│   ├── llm/                     # local / api LLM runtime
│   ├── ml/                      # reserved
│   └── rom/                     # reserved
├── artifacts/models/            # 真实模型权重放这里，不放进 Python package
│   ├── llm/
│   ├── embedding/
│   └── reranker/
├── data/documents/              # CAE 文档
├── storage/                     # Chroma / cache / manifest
├── scripts/
├── tests/
├── .env.example
└── requirements.txt
```

---

# 4. 推荐模型

## LLM

第一版建议先使用：

- `Qwen/Qwen3-8B`

模型地址：

- https://huggingface.co/Qwen/Qwen3-8B

如果显存不足，可换更小的 Qwen 模型；如果后续需要更高吞吐，建议再把本地 LLM runtime 从直接 Transformers 替换成 vLLM/SGLang，而不是改 RAG 业务层。

## Embedding

第一版建议：

- `BAAI/bge-m3`

模型地址：

- https://huggingface.co/BAAI/bge-m3

BGE-M3 对中文、多语言、长文本都比较适合，而且后续可以继续扩展 sparse/multi-vector 能力。

## Optional Reranker

- `BAAI/bge-reranker-v2-m3`

模型地址：

- https://huggingface.co/BAAI/bge-reranker-v2-m3

第一版可以先不开 reranker，确认 Dense RAG 正常后再启用。

---

# 5. 在联网电脑下载模型

先安装 Hugging Face CLI：

```bash
pip install -U huggingface_hub
```

创建目录：

```bash
mkdir -p artifacts/models/llm
mkdir -p artifacts/models/embedding
mkdir -p artifacts/models/reranker
```

下载 LLM：

```bash
hf download Qwen/Qwen3-8B \
  --local-dir artifacts/models/llm/Qwen3-8B
```

下载 Embedding：

```bash
hf download BAAI/bge-m3 \
  --local-dir artifacts/models/embedding/bge-m3
```

可选下载 Reranker：

```bash
hf download BAAI/bge-reranker-v2-m3 \
  --local-dir artifacts/models/reranker/bge-reranker-v2-m3
```

然后把整个 `artifacts/models/` 拷到离线设备对应目录。

**不要只复制单个 `.safetensors`。** tokenizer、config、sentence-transformers 模块配置等文件也需要一起带走，所以推荐直接下载整个 repo snapshot。

---

# 6. 离线设备安装 Python 环境

建议 Python 3.10 或 3.11。

## 6.1 创建环境

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
```

Windows：

```powershell
python -m venv .venv
.venv\Scripts\activate
```

## 6.2 PyTorch

如果有 NVIDIA GPU，请在联网机器上根据你离线设备的 CUDA/驱动环境准备对应 PyTorch wheel。**不要盲目使用 requirements.txt 自动决定 CUDA 版本。**

确认 PyTorch：

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

## 6.3 其他依赖

联网设备可以提前下载 wheel：

```bash
pip download -r requirements.txt -d wheelhouse
```

把 `wheelhouse/` 带到离线设备后：

```bash
pip install --no-index --find-links=wheelhouse -r requirements.txt
```

如果你的 PyTorch wheel 已单独安装，pip 会复用已有版本。

---

# 7. 配置

复制配置：

```bash
cp .env.example .env
```

默认已经是本地模式：

```env
CAE_LLM_PROVIDER=local
CAE_LLM_MODEL_PATH=./artifacts/models/llm/Qwen3-8B

CAE_EMBED_PROVIDER=local
CAE_EMBED_MODEL_PATH=./artifacts/models/embedding/bge-m3

CAE_RETRIEVAL_MODE=dense
CAE_RERANK_ENABLED=false
```

`.env` 会自动读取，不需要再手动 `export`。

---

# 8. 第一次启动：严格按这个顺序

## Step 1：检查环境与模型目录

```bash
python scripts/check_models.py
```

至少应该看到：

```text
chromadb               : OK
torch                   : OK
transformers            : OK
sentence_transformers   : OK
LLM path                : ... [OK]
Embedding path          : ... [OK]
```

## Step 2：做本地模型 smoke test

```bash
python scripts/smoke_test.py
```

它会测试：

1. Embedding 是否能真正生成向量；
2. Chroma 是否能启动；
3. LLM 是否能真正生成文本。

## Step 3：放入 CAE 文档

把资料放到：

```text
data/documents/
```

支持：

```text
.pdf
.docx
.pptx
.txt
.md
```

## Step 4：建立知识库

第一次：

```bash
python scripts/ingest.py data/documents --recreate
```

以后文档更新：

```bash
python scripts/ingest.py data/documents
```

程序会自动跳过内容没有变化的文件。

如果强制重新编码：

```bash
python scripts/ingest.py data/documents --force
```

如果 **Embedding 模型换了**，一定重新建库：

```bash
python scripts/ingest.py data/documents --recreate
```

---

# 9. 查询

## 推荐：常驻服务，避免每次提问加载权重

直接运行 `python scripts/ask.py "问题"` 会创建新的 Python 进程，回答后退出并释放模型；下次运行必须重新加载。模型文件已经在磁盘上，不代表权重仍驻留在内存/显存中。

先在一个终端启动服务并保持运行：

```bash
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --workers 1
```

然后在另一个终端提问：

```bash
python scripts/ask.py "缸盖热应力分析中的热边界条件是什么？" --server http://127.0.0.1:8000
python scripts/ask.py "如何设置约束？" --server http://127.0.0.1:8000
python scripts/ask.py "温度边界条件" --server http://127.0.0.1:8000 --no-gen --top-k 3
```

`--server` 连接本项目的 CAE API，客户端不加载 LLM、Embedding 或本地知识库。服务在首次需要时加载模型，后续问题复用同一 pipeline 中的模型；未检索到资料时不会加载 LLM。问题仍独立回答，不保存多轮对话历史。

保持单 worker，且不要使用 `--reload`：多个 worker 各自加载模型，自动重启会重新加载。服务停止或电脑重启后，下一次仍需要加载一次。首次加载较慢，CLI 请求超时为 600 秒；请求失败会报错退出，不会退回本地加载。原来的不带 `--server` 的本地单次模式继续可用。

## 只看检索结果

建议先这样调 RAG：

```bash
python scripts/ask.py "缸盖热应力分析中的热边界条件是什么？" --no-gen
```

## 检索 + LLM

```bash
python scripts/ask.py "缸盖热应力分析中的热边界条件是什么？"
```

---

# 10. 启动 Web/API

```bash
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

浏览器打开：

```text
http://127.0.0.1:8000
```

API：

```text
GET  /api/health
POST /api/search
POST /api/query
POST /api/ingest
```

---

# 11. 开启 Hybrid Retrieval

编辑 `.env`：

```env
CAE_RETRIEVAL_MODE=hybrid
CAE_CANDIDATE_K=20
CAE_TOP_K=6
```

当前 Hybrid：

```text
Dense vector search
        +
Local BM25
        ↓
Reciprocal Rank Fusion (RRF)
        ↓
Top candidates
```

对“材料牌号、零件编号、软件关键词、载荷工况、数值边界条件”这类 CAE 查询尤其值得测试。

---

# 12. 开启 Reranker

确保模型已放到：

```text
artifacts/models/reranker/bge-reranker-v2-m3/
```

配置：

```env
CAE_RERANK_ENABLED=true
CAE_RERANK_MODEL_PATH=./artifacts/models/reranker/bge-reranker-v2-m3
CAE_CANDIDATE_K=20
CAE_TOP_K=6
```

推荐链路：

```text
Hybrid retrieve 20 chunks
        ↓
Reranker
        ↓
Keep top 6
        ↓
LLM
```

Reranker 通常提升排序质量，但会增加延迟。因此默认关闭。

---

# 13. 切换到 API

## LLM API

```env
CAE_LLM_PROVIDER=api
CAE_LLM_BASE_URL=http://your-server:8000/v1
CAE_LLM_API_KEY=xxx
CAE_LLM_MODEL=your-model-name
```

## Embedding API

```env
CAE_EMBED_PROVIDER=api
CAE_EMBED_BASE_URL=http://your-server:8001/v1
CAE_EMBED_API_KEY=xxx
CAE_EMBED_MODEL=your-embedding-name
```

只要服务是 OpenAI-compatible，RAG 层不用修改。

---

# 14. 性能调优建议

## Embedding 慢

优先调整：

```env
CAE_EMBED_BATCH=64
CAE_EMBED_MAX_LENGTH=512
```

如果文档 chunk 本身并不长，没有必要让 embedding 每次处理 8192 token。

## LLM 显存不足

可尝试：

```env
CAE_LLM_DTYPE=float16
CAE_LLM_DEVICE_MAP=auto
```

真正需要 4bit/AWQ/GPTQ 时，建议增加专门 runtime provider，而不是把量化判断塞进 RAG pipeline。

## 检索慢

先使用：

```env
CAE_RETRIEVAL_MODE=dense
CAE_CANDIDATE_K=10
CAE_RERANK_ENABLED=false
```

精度不够再开启 Hybrid / Reranker。

## 大规模知识库

当前 Chroma + 内存 BM25 更适合部门级/项目级知识库。若未来达到几十万到百万 chunk，再考虑：

- Qdrant / Milvus / Elasticsearch/OpenSearch
- 服务化 embedding
- GPU batch embedding
- 原生 sparse + dense hybrid index
- query cache
- reranker service

接口已经分层，未来替换 vector store/retriever 不需要重写 Loader、Chunker、Prompt、LLM。

---

# 15. 测试

运行单元测试：

```bash
pytest -q
```

语法检查：

```bash
python -m compileall -q core knowledge models agents domains app scripts tests
```

离线真实运行测试：

```bash
python scripts/check_models.py
python scripts/smoke_test.py
python scripts/ingest.py data/documents --recreate
python scripts/ask.py "测试问题" --no-gen
python scripts/ask.py "测试问题"
```

---

# 16. 目前没有做、但适合下一阶段做的能力

- Query rewrite / Multi-query retrieval
- Metadata filters（强度/NVH/CFD、车型、零件、项目、版本）
- Parent-child chunk / sentence-window retrieval
- BGE-M3 原生 sparse / multi-vector 检索
- RAG 自动评测集与 Recall@K / MRR / nDCG
- 文档版本治理与权限控制
- 表格专用检索
- 图片/曲线/CAE 云图多模态 RAG
- vLLM/SGLang 本地 LLM 服务化
- NVH / Strength / CFD domain router
- CAE Agent 调求解器、后处理和报告生成

建议下一阶段优先做：**RAG evaluation + metadata + parent-child chunking**，而不是先堆 Agent。
