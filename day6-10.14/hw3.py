# 作业 3（进阶）：把 ask_ollama 改成流式（复用 Day4 的代码），
#   对比首字延迟
# 流式脚本版框架
import math
import time
import httpx
import json

OLLAMA_URL = "http://127.0.0.1:11434"
CHAT_MODEL = "qwen2.5-1.5b"
EMBED_MODEL = "bge-m3"


def split_text(text, chunk_size=200, overlap=50):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        if end >= len(text):
            break
        start = end - overlap
    return chunks


def get_embeddings(texts):
    resp = httpx.post(
        OLLAMA_URL + "/api/embed",
        json={"model": EMBED_MODEL, "input": texts},
        timeout=300,
    )
    resp.raise_for_status()
    return resp.json()["embeddings"]


def cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb)


def retrieve(question, chunk_vecs, top_k=3):
    q_vec = get_embeddings([question])[0]
    scores = []
    for i, v in enumerate(chunk_vecs):
        scores.append((cosine(v, q_vec), i))
        scores.sort(reverse=True)
    return scores[:top_k]


def build_prompt(question, chunks, top):
    """把检索到的块拼成给模型的 prompt"""

    # 1.约束 instruction Python中 ()里的代码可以跨行 并且相邻的字符串不用 + 也能拼接
    instruction = (
        "请严格根据下面的资料回答问题"
        "如果资料里没有答案 就回答'资料中没有找到答案'"
        "回答时请标注引用编号 如 [1]"
    )

    # 2.资料 context
    context_parts = []
    for rank, (_, i) in enumerate(top, start=1):
        context_parts.append(f"[{rank}] {chunks[i]}")
    context = "\n\n".join(context_parts)
    """
    _ 表示不需要这个值 即 top 中每个元组的 score 不需要
    i 经排序后已经乱了 要用 rank
    enumerate(top, start=1)
    会产生 一个元组 (1, top[0]) 自动加上编号
    start = 1 表示 编号从 1 开始 没有参数就是从 0 开始
    再把上面生成的元组拆开 分别赋给 rank 和 (_, i)
    "\n\n"就是分隔符 分隔符.join(列表) 把列表里的每个字符串用分隔符拼起来
    如 "\n\n".join(["A", "B", "C"])
    结果 A\n\nB\n\nC
    """

    # 3.拼 prompt = instruction + context + question
    prompt = (
        f"{instruction}\n\n"
        f"资料:\n{context}\n\n"
        f"问题: {question}\n"
        f"回答: "
    )

    return prompt


def ask_ollama(prompt):
    """流式调用，返回 (完整回答, TTFT, 总耗时)"""
    full_reply = []
    start = time.time()
    first_token_time = None
    with httpx.stream(
        "POST",
        OLLAMA_URL + "/api/chat",
        json={
            "model": CHAT_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "stream": True,
        },
        timeout=300,
    ) as resp:
        resp.raise_for_status()
        for line in resp.iter_lines():
            if not line:
                continue # 跳过空行
            data  = json.loads(line)
            # 此处要记住 Ollama 返回的 JSON 格式
            piece = data.get("message", {}).get("content", "")
            if piece:
                if first_token_time is None:
                    first_token_time = time.time() - start
                full_reply.append(piece)
                print(piece, end="", flush=True)
    total = time.time() - start
    return "".join(full_reply), first_token_time, total


if __name__ == "__main__":
    with open("sample.txt", "r", encoding="utf-8") as f:
        text = f.read()

    # 建库：切分 + 批量向量化
    chunks = split_text(text)
    chunk_vecs = get_embeddings(chunks)
    print("知识库就绪：", len(chunks), "块")

    question = input("请输入问题：")

    # ① 检索（计时）
    t0 = time.time()
    top = retrieve(question, chunk_vecs, top_k=3)
    retrieval_ms = (time.time() - t0) * 1000

    # ② 拼 prompt + 生成（计时）
    prompt = build_prompt(question, chunks, top)
    answer, first_token_time, total = ask_ollama(prompt)

    # ③ 输出
    print("\n=== 回答 ===")
    print(answer)

    print("\n=== 引用来源 ===")
    for rank, (score, i) in enumerate(top, start=1):
        print(f"[{rank}] chunk[{i}] 相似度 {score:.4f}:{chunks[i][:50]}...")

    print(f"\n检索耗时 {retrieval_ms:.0f} ms | TTFT {first_token_time:.0f}ms | 生成耗时 {total:.0f} ms")
