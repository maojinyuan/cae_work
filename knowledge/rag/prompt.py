from __future__ import annotations

SYSTEM_PROMPT = """你是一名资深的发动机 CAE 仿真专家助手，擅长结构强度、热分析、疲劳、CFD、NVH、多体动力学等领域。

请严格遵循：
1. 只依据【参考资料】作答，不要使用资料之外的知识臆测。
2. 引用资料时使用 [1][2] 等编号，并在最后列出参考来源。
3. 资料不足时明确说明“资料中未找到相关依据”，不要编造数据或结论。
4. CAE 数值、边界条件、材料参数、单位、公式原样保留，不擅自换算。
5. 用中文回答，专业术语和软件名可保留英文。
6. 面向工程师，先给结论，再给依据和关键参数。
"""


def build_messages(question: str, contexts) -> list[dict]:
    refs = []
    for i, c in enumerate(contexts, 1):
        loc = c.source
        if c.page:
            loc += f" 第{c.page}页"
        if c.section:
            loc += f" · {c.section}"
        tag = "[表格] " if c.kind == "table" else ""
        refs.append(f"[{i}] {tag}来源:{loc}\n{c.text}")
    reference_text = "\n\n".join(refs)
    user = f"【问题】\n{question}\n\n【参考资料】\n{reference_text}\n\n请依据以上资料回答【问题】。"
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]


def format_sources(contexts):
    return [{"index": i, "source": c.source, "page": c.page, "section": c.section, "kind": c.kind, "score": c.score, "text": c.text} for i, c in enumerate(contexts, 1)]
