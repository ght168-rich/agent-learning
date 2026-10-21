import json
import math
import time
import jieba
import re
import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

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


def build_prompt(question, top):
    """约束 + 编号资料 + 问题 Day6 写过，直接复用"""
    instruction = (
        "请严格根据下面的资料回答问题。"
        "如果资料中没有答案，就回答'资料中没有找到'。"
        "回答时请标注引用编号，比如 [1]。"
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
