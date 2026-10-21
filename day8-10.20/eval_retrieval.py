import json
import math
import re
import httpx
import jieba

OLLAMA_URL = "http://127.0.0.1:11434"
EMBED_MODEL = "bge-m3"


# 文本切片
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

    print(f"文本块数量 {len(chunks)}")
    for i, c in enumerate(chunks[:5]):
        print(f"--- chunk {i} (len={len(c)}) ---")
        print("开头:", c[:40])
        print("结尾:", c[-40:])

    return chunks
"""
原本的按字符数量切分的方式会中断完整句子
即使模型检索到了也没法用 会降低回答质量
现在按句子/段落切分 保留重叠
"""


# 文本向量化
def get_embeddings(texts):
    resp = httpx.post(
        OLLAMA_URL + "/api/embed",
        json={"model": EMBED_MODEL, "input": texts},
        timeout=300,
    )
    resp.raise_for_status()
    return resp.json()["embeddings"]


# 计算余弦相似度
def cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    # math.sqrt() 开平方
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb)


# jieba 切词
def tokenize(text):
    """
    jieba 分词，只保留长度 >= 2 的词
    jieba.lcut() 用 jieba 切词 返回一个列表
    """
    return [w for w in jieba.lcut(text) if len(w.strip()) >= 2]


# 计算关键词在 chunk 里命中的比例
def keyword_score(question, chunk):
    """问题关键词在 chunk 里命中的比例 0~1"""
    # 把 jieba 分词后的列表转换成集合 集合可以求交集并自动去重
    q_tokens = set(tokenize(question))
    c_tokens = set(tokenize(chunk))
    if not q_tokens:
        return 0.0
    # & 取两个集合的交集 取问题和文本块的交集 算命中词数在问题中的比例
    return len(q_tokens & c_tokens) / len(q_tokens)


# 取纯向量检索余弦相似度倒数的 k 个文本块
def retrieve_vector(question, chunks, vecs, k=3):
    q_vec = get_embeddings([question])[0]
    scores = [(cosine(q_vec, v), i) for i, v in enumerate(vecs)]
    scores.sort(reverse=True)
    return scores[:k]


# 混合检索 把向量检索得分和关键词检索得分按一定权重相加
def retrieve_hybrid(question, chunks, vecs, k=3, alpha=0.7):
    q_vec = get_embeddings([question])[0]
    scores = []
    for i, v in enumerate(vecs):
        vs = cosine(q_vec, v)
        ks = keyword_score(question, chunks[i])
        # 把两个分数加权相加：alpha 是向量权重
        combined = alpha * vs + (1 - alpha) * ks
        scores.append((combined, i))
    scores.sort(reverse=True)
    return scores[:k]


def evaluate(retrieve_fn, eval_set, chunks, k=3):
    """针对两种检索方式 计算 Recall@K 和 MRR"""
    # 有多少问题检索命中
    hits = 0
    # 每个正确结果排名的得分之和
    mrr = 0.0

    # 遍历 10 个问题
    for item in eval_set:
        # retrieve_fn 是检索函数 可以传纯向量或者混合检索
        # 返回得分倒数 k 个的 (得分, 编号) 元组
        top = retrieve_fn(item["question"])

        rank_hit = None
        for rank, (_, idx) in enumerate(top, start=1):
            # any() 循环遍历中 有一次循环满足前面的条件即为真
            if any(kw in chunks[idx] for kw in item["keywords"]):
                # 记录下命中的 rank 编号
                # rank 是外层 for 循环拆包时给的编号
                # 即在那倒数k个文本块中的排名
                rank_hit = rank
                break

        if rank_hit is not None:
            # 命中问题数 +1
            hits += 1
            # 排名取倒数加入总得分
            mrr += 1.0 / rank_hit

    n = len(eval_set)
    # 取平均值返回
    return hits / n, mrr / n


if __name__ == "__main__":
    with open("sample.txt", "r", encoding="utf-8") as f:
        text = f.read()
    chunks = split_text(text)
    vecs = get_embeddings(chunks)

    with open("eval_set.json", "r", encoding="utf-8") as f:
        eval_set = json.load(f)

    """
    匿名函数 lambda 参数: 表达式
    函数需要传多个参数 但是调用时我只传一个 用 lambda 把其他参数固定住
    只留语法中出现的那个参数在调用时传
    top = retrieve_fn(item["question"]) 中只传了一个参数
    """
    recall_v, mrr_v = evaluate(
        lambda q: retrieve_vector(q, chunks, vecs, 3), eval_set, chunks, 3
    )
    recall_h, mrr_h = evaluate(
        lambda q: retrieve_hybrid(q, chunks, vecs, 3, 0.7), eval_set, chunks, 3
    )

    print(f"纯向量检索  Recall@3 = {recall_v:.4f} MRR = {mrr_v:.4f}")
    print(f"混合检索    Recall@3 = {recall_h:.4f} MRR = {mrr_h:.4f}")
    """
    实验数据记录
    方法------Recall@3------MRR
    纯向量检索------0.9------0.9
    混合检索(alpha=0.7)------1.0------1.0
    混合检索(alpha=0.5)------1.0------1.0
    混合检索(alpha=0.9)------1.0------1.0
    """