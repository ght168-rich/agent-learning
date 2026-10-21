# 实现最小检索器
import math
import httpx

OLLAMA_URL = "http://127.0.0.1:11434"
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


# 实现最小检索器
if __name__ == "__main__":
    with open("sample.txt", "r", encoding="utf-8") as f:
        text = f.read()

    chunks = split_text(text)
    print("切出", len(chunks), "块")

    # 1. 把所有块批量变成向量（一次请求，快）
    chunk_vecs = get_embeddings(chunks)

    # 2. 输入问题，变成向量
    question = input("请输入你的问题：")
    # 传入列表 列表只有一个 question 元素
    q_vec = get_embeddings([question])[0] # 返回的是向量列表 要取第一个

    scores = []
    for i, v in enumerate(chunk_vecs):
        scores.append((cosine(v, q_vec), i))
        scores.sort(reverse=True)
    for score, i in scores[:3]:
        print(f"相似度{score:.4f}:{chunks[i][:80]}...")
    print("检索完成")

import time

# 方式 A：批量（一次请求）
start = time.time()
_ = get_embeddings(chunks) # _= 拿到结果但不需要它
print("批量耗时：", round(time.time() - start, 2), "秒")

# 方式 B：逐条（N 次请求）
start = time.time()
# get_embeddings 输入输出都是列表 所以要写 [c] [0]
_ = [get_embeddings([c])[0] for c in chunks]
"""
等价代码
results = []
for c in chunks:
    results.append(get_embeddings([c])[0])
_ = results
"""
print("逐条耗时：", round(time.time() - start, 2), "秒")
"""
批量耗时 0.39 秒
逐条耗时 1.51 秒
块数越多 批量比逐条快得越明显
"""
