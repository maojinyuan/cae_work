# Code Review & Refactor Notes

## 原版本主要问题

1. **Provider 层不完全闭合**：factory 已存在，但 package `__init__` 没有稳定导出，直接运行 CLI 时会触发 ImportError。
2. **FastEmbed 与 SentenceTransformers 重叠**：同一项目维护两套本地 embedding 运行时会增加模型兼容、缓存和依赖管理成本；现阶段删除 FastEmbed 主路线。
3. **配置行为不一致**：README 写法与代码默认值不一致；`.env` 需要手工 export，容易在离线部署中出错。
4. **RAG 只有单路 Dense Top-K**：缺少候选扩召回、Hybrid、Reranker、缓存等常见升级点。
5. **重复入库成本高**：文档未变化也会重新 embedding。
6. **Embedding 重复计算**：相同 chunk 没有持久化缓存。
7. **大批量写库不分批**：知识库扩大后内存和写入压力会增加。
8. **CLI 可执行性问题**：直接 `python scripts/*.py` 时项目根目录不一定在 `sys.path`。
9. **API key 约束过严**：vLLM/SGLang 等内网 OpenAI-compatible 服务通常不需要真实 key。
10. **本地 LLM 与特定模型特性未隔离**：Qwen3 thinking 模式可能导致 RAG 延迟和输出 `<think>`；新增统一开关，默认关闭。

## 当前版本的选择

- 默认 Local LLM + Local Embedding。
- API 作为同接口 provider。
- Dense 默认，Hybrid/Reranker 按需开启。
- Chroma 暂保留，适合当前阶段；不急于引入 Milvus/Qdrant。
- Embedding 使用 SentenceTransformers 统一运行层。
- 真实权重统一放 `artifacts/models/`，与 Python `models/` 代码包分离。

## 下一阶段优先级

1. RAG 评测集（Recall@K / MRR / nDCG / answer faithfulness）
2. Metadata schema 与过滤
3. Parent-child chunking / sentence-window retrieval
4. BGE-M3 原生 sparse + dense hybrid
5. 文档版本与权限
6. 表格/图片/曲线多模态 RAG
7. vLLM/SGLang provider
8. CAE Agent / solver tools
