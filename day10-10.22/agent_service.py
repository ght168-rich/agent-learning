import json
import math
import time
import jieba
import re
import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from datetime import datetime

OLLAMA_URL = "http://127.0.0.1:11434"
CHAT_MODEL = "qwen2.5-1.5b"
EMBED_MODEL = "bge-m3"
DATA_FILE = "sample.txt"

app = FastAPI(title="RAG Agent Service")

# ============ 全局索引：启动时建一次 ============
INDEX = {"chunks": [], "vecs": [], "token_sets": []}


def split_text(text, chunk_size=200, overlap_sentences=1):
    # 按中/英文句末标点和换行切句 保留标点
    sentences = [
        s.strip()
        for s in re.split(r'(?<=[。！？!?；;\n])',text)
        if s.strip()
    ]

    chunks = []
    current = []
    length = 0

    for s in sentences:
        if length + len(s) > chunk_size and current:
            chunks.append("".join(current))
            # 用上一块最后 overlap_sentences 句作为下一块开头 制造语义重叠
            current = current[-overlap_sentences:]
            # 更新 current 之后更新 length
            length = sum(len(x) for x in current)

        current.append(s)
        length += len(s)

    # 把最后一块也加进列表
    if current:
        chunks.append("".join(current))

    return chunks
"""
原本的按字符数量切分的方式会中断完整句子
即使模型检索到了也没法用 会降低回答质量
现在按句子/段落切分 保留重叠
"""


def get_embeddings(texts):
    """建索引用（同步、批量，启动时执行一次）"""
    resp = httpx.post(
        OLLAMA_URL + "/api/embed",
        json={"model": EMBED_MODEL, "input": texts},
        timeout=300,
    )
    resp.raise_for_status()
    return resp.json()["embeddings"]


async def embed_query(text):
    """请求路径上用（异步，单个问题）"""
    async with httpx.AsyncClient(timeout=120) as client:
        resp = await client.post(
            OLLAMA_URL + "/api/embed",
            json={"model": EMBED_MODEL, "input": text},
        )
        resp.raise_for_status()
        return resp.json()["embeddings"][0]


def cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb)


def tokenize(text):
    return [w for w in jieba.lcut(text) if len(w.strip()) >= 2]


def keyword_score(q_tokens, chunk_tokens):
    if not q_tokens:
        return 0.0
    return len(q_tokens & chunk_tokens) / len(q_tokens)


def build_index(path=DATA_FILE):
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    chunks = split_text(text)
    vecs = get_embeddings(chunks)
    # token_sizes 是列表里装着集合
    # 对每个文本块进行分词 然后存进集合 每个文本块一个集合
    token_sets = [set(tokenize(c)) for c in chunks]
    return chunks, vecs, token_sets


@app.on_event("startup")
def startup():
    """启动时只建一次索引"""
    chunks, vecs, token_sets = build_index()
    INDEX["chunks"] = chunks
    INDEX["vecs"] = vecs
    INDEX["token_sets"] = token_sets
    print(f"[startup] 索引构建完成：{len(chunks)} 块")


async def retrieve(question, top_k=3, alpha=0.7):
    q_vec = await embed_query(question)
    q_tokens = set(tokenize(question))
    scores = []
    for i, v in enumerate(INDEX["vecs"]):
        # 算复合检索分数 存成 (分数, 下标)
        vs = cosine(v, q_vec)
        ks = keyword_score(q_tokens, INDEX["token_sets"][i])
        combine = alpha * vs + (1 - alpha) * ks
        scores.append((combine, i))
    scores.sort(reverse=True)
    return scores[:top_k]


def tool_get_current_time():
    """返回当前日期和时间"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def tool_calculate(expression: str):
    """安全计算数学表达式，只允许数字和 + - * / ( ) ."""
    if not re.fullmatch(r"[0-9+\-*/(). ]+", expression):
        return "表达式非法，只支持数字和 + - * / ( )"
    try:
        return str(eval(expression, {"__builtins__": {}}, {}))
    except Exception as e:
        return f"计算失败: {e}"


async def tool_search_knowledge(query: str, top_k: int = 3):
    """检索本地知识库，返回 Top-K 文本块"""
    top = await retrieve(query, top_k=top_k, alpha=0.7)
    return "\n\n".join(
        f"[{r + 1}] {INDEX['chunks'][i]}"
        for r, (score, i) in enumerate(top)
    )    


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "获取当前日期和时间",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "计算数学表达式，例如 123*45+6",
            "parameters": {
                "type": "object",
                "properties": {"expression": {"type": "string"}},
                "required": ["expression"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_knowledge",
            "description": (
                "当问题涉及 RAG、知识库、项目文档、课程资料或本地资料时，"
                "必须调用此工具检索；不要凭模型自身知识回答。"
            ),
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        },
    },
]


async def chat_with_tools(messages):
    async with httpx.AsyncClient(timeout=300) as client:
        resp = await client.post(
            OLLAMA_URL + "/api/chat",
            json={
                "model": CHAT_MODEL,
                "messages": messages,
                "tools": TOOLS,
                "stream": False,
                "options": {"temperature": 0},
            },
        )
        resp.raise_for_status()
        return resp.json().get("message", {})


async def execute_tool(name, args):
    if name == "get_current_time":
        return tool_get_current_time()
    if name == "calculate":
        return tool_calculate(args.get("expression", ""))
    if name == "search_knowledge":
        return await tool_search_knowledge(args.get("query", ""))
    return f"未知工具: {name}"    


def plan_required_tools(question: str):
    """根据问题，列出需要调用的工具"""
    plan = []
    q = question.lower()

    # 时间类
    if any(k in q for k in ["几点", "时间", "日期", "今天"]):
        plan.append(("get_current_time", {}))

    # 数学表达式（找带运算符的那一段）
    for m in re.findall(r"[0-9][0-9+\-*/().\s]*", question):
        expr = m.strip()
        if re.search(r"[+\-*/]", expr):
            plan.append(("calculate", {"expression": expr}))
            break

    # 知识库类
    if any(k in q for k in ["rag", "知识库", "文档", "资料", "项目", "检索"]):
        plan.append(("search_knowledge", {"query": question}))

    return plan


def build_prompt(question, top):
    """约束 + 编号资料 + 问题 Day6 写过，直接复用"""
    instruction = (
        "请严格根据下面的资料回答问题。"
        "如果资料中没有答案，就回答'资料中没有找到'。"
        "回答时请标注引用编号，比如 [1]。"
        "回答控制在 3 句话以内。"
    )
    context_parts = []
    for rank, (_, i) in enumerate(top, start=1):
        # 把 INDEX["chunks"][i] 按 "[rank] 内容" 拼进去
        context_parts.append(f"[{rank}] {INDEX["chunks"][i]}")
    context = "\n\n".join(context_parts)
    return f"{instruction}\n\n资料：\n{context}\n\n问题：{question}\n回答："


# ============ 非流式调用模型 ============
async def ask_ollama(prompt):
    start = time.time()
    async with httpx.AsyncClient(timeout=300) as client:
        resp = await client.post(
            OLLAMA_URL + "/api/chat",
            json={
                "model": CHAT_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
                "options": {"temperature": 0},
            },
        )
        resp.raise_for_status()
        reply = resp.json()["message"]["content"]
    return reply, (time.time() - start) * 1000


# ============ 流式调用模型 ============
async def stream_ollama(prompt):
    """异步生成器 async for 逐行读 → yield SSE"""
    async with httpx.AsyncClient(timeout=300) as client:
        async with client.stream(
            "POST",
            OLLAMA_URL + "/api/chat",
            json={
                "model": CHAT_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "stream": True,
                "options": {"temperature": 0},
            },
            timeout=300,
        ) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                data = json.loads(line)
                piece = data.get("message", {}).get("content", "")
                if piece:
                    yield "data: " + json.dumps({"text": piece}, ensure_ascii=False) + "\n\n"
    yield 'data: {"done": true}\n\n'


def route_question(question: str):
    """模型没调工具时，按关键词/正则选择工具"""
    if any(k in question for k in ["几点", "时间", "日期", "今天"]):
        return "get_current_time", {}

    m = re.search(r"[0-9][0-9+\-*/().\s]*", question)
    if m and re.search(r"[+\-*/]", m.group(0)):
        return "calculate", {"expression": m.group(0).strip()}

    return "search_knowledge", {"query": question}


async def run_agent(question, max_steps=3):
    plan = plan_required_tools(question)
    trace = []

    # ===== 多意图：先把所有必需工具执行完，再让模型统一总结 =====
    if len(plan) > 1:
        messages = [
            {"role": "system", "content": (
                "你是助手。必须先使用所有工具返回的结果；"
                "涉及计算时，直接采用 calculate 的结果，不得自行计算。"
            )},
            {"role": "user", "content": question},
        ]

        for step, (name, args) in enumerate(plan, start=1):
            result = await execute_tool(name, args)
            trace.append({
                "step": step,
                "tool": name,
                "args": args,
                "result": str(result)[:300],
            })
            messages.append({
                "role": "assistant",
                "content": "",
                "tool_calls": [{"function": {"name": name, "arguments": args}}],
            })
            messages.append({
                "role": "tool",
                "content": str(result),
                "tool_name": name,
            })

        final = await chat_with_tools(messages)
        return final.get("content", ""), trace

    # ===== 单意图：保留你原来的模型决策循环 =====
    messages = [
        {"role": "system", "content": "你是一个会使用工具的助手。需要工具时调用工具，否则直接回答。"},
        {"role": "user", "content": question},
    ]

    for step in range(1, max_steps + 1):
        msg = await chat_with_tools(messages)
        calls = msg.get("tool_calls") or []

        if not calls and step == 1:
            name, args = route_question(question)
            msg = {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"function": {"name": name, "arguments": args}}],
            }
            calls = msg["tool_calls"]

        messages.append(msg)

        if not calls:
            return msg.get("content", ""), trace

        for call in calls:
            fn = call.get("function", {})
            name = fn.get("name", "")
            args = fn.get("arguments", {})
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {}

            result = await execute_tool(name, args)
            trace.append({
                "step": step,
                "tool": name,
                "args": args,
                "result": str(result)[:300],
            })
            messages.append({
                "role": "tool",
                "content": str(result),
                "tool_name": name,
            })

    return "达到最大步数，未能完成", trace


# ============ 请求/响应模型 ============
class AskRequest(BaseModel):
    question: str
    top_k: int = 3
    alpha: float = 0.7


class AskResponse(BaseModel):
    answer: str
    sources: list
    retrieval_ms: float
    generation_ms: float


# ============ 接口 ============
@app.post("/ask", response_model=AskResponse)
async def ask(req: AskRequest):
    """检索 → 拼 prompt → 调模型 → 返回答案 + 来源 + 耗时"""
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="question 不能为空")

    t0 = time.time()
    top = await retrieve(req.question, req.top_k, req.alpha)
    retrieval_ms = (time.time() - t0) * 1000

    prompt = build_prompt(req.question, top)
    answer, gen_ms = await ask_ollama(prompt)

    sources = [
        {"rank": r + 1, "score": round(s, 4), "text": INDEX["chunks"][i][:60]}
        for r, (s, i) in enumerate(top)
    ]
    return AskResponse(
        answer=answer, sources=sources,
        retrieval_ms=round(retrieval_ms, 1),
        generation_ms=round(gen_ms, 1),
    )


@app.post("/ask/stream")
async def ask_stream(req: AskRequest):
    """检索 + 拼 prompt + 返回 StreamingResponse"""
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="question 不能为空")

    top = await retrieve(req.question, req.top_k, req.alpha)
    prompt = build_prompt(req.question, top)

    return StreamingResponse(
        stream_ollama(prompt),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache"},
    )


class AgentRequest(BaseModel):
    question: str


@app.post("/agent")
async def agent(req: AgentRequest):
    start = time.time()
    answer, trace = await run_agent(req.question)
    return {
        "answer": answer,
        "trace": trace,
        "latency_ms": round((time.time() - start) * 1000, 2),
    }


if __name__ == "__main__":
    print("时间工具：", tool_get_current_time())
    print("计算器：", tool_calculate("123 * 45 + 6"))
    print("非法表达式：", tool_calculate("__import__('os').system('dir')"))