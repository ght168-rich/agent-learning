import math
import time
import httpx

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
    """调用对话模型，返回 (回答文本, 耗时毫秒)"""
    start = time.time()
    resp = httpx.post(
        OLLAMA_URL + "/api/chat",
        json={
            "model": CHAT_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
        },
        timeout=300,
    )
    resp.raise_for_status()
    reply = resp.json()["message"]["content"]
    return reply, (time.time() - start) * 1000


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
    answer, gen_ms = ask_ollama(prompt)

    # ③ 输出
    print("\n=== 回答 ===")
    print(answer)

    print("\n=== 引用来源 ===")
    for rank, (score, i) in enumerate(top, start=1):
        print(f"[{rank}] chunk[{i}] 相似度 {score:.4f}:{chunks[i][:50]}...")

    print(f"\n检索耗时 {retrieval_ms:.0f} ms | 生成耗时 {gen_ms:.0f} ms")
"""
top_k = 3
检索耗时 123ms
生成耗时 2444ms
回答 检索增强生成技术 即RAG 是一种为了解决训练模型知识固定 不知道最新事情以及回答问题时可能产生幻觉等问题的技术
"""
# 作业 1：把 top_k 改成 1 和 5，各测一次
#   观察：回答质量有什么变化？检索耗时和生成耗时怎么变？
"""
top_k = 1
检索耗时 156ms
生成耗时 1894ms
回答 检索增强生成技术 RAG 是一种技术 它通过将大语言模型的参数固定在训练完成后
使其无法了解公司内部文档 项目资料和个人笔记 同时确保其知识具有截止日期 且当模型不知道答案时会编造看起来合理但实际错误的内容
其核心思想是通过检索来增强模型的知识
"""
"""
top_k = 5
检索耗时 134ms
生成耗时 1569ms
回答 检索增强生成技术 也就是常说的 RAG 正是为了解决这三个问题而出现的
"""
"""
对性能调优的启示
1.检索不是瓶颈 生成才是瓶颈
2.top_k 表示给到模型资料的多少 资料多不一样回答好
3.多召回 + 精排 先检索20块 再 rerank 出最相关的 3 块
4.流式输出 降低延迟
5.减少重复计算 问题->向量 的结果可以缓存 相同问题直接命中
"""

# 作业 2：不问文章里的内容（比如问"今天上海天气"），
#   验证模型是否老实回答"资料中没有找到"
"""
模型老实回答
"""

# 作业 3（进阶）：把 ask_ollama 改成流式（复用 Day4 的代码），
#   对比首字延迟