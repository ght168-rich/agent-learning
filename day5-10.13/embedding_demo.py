# 实现文本向量化 并计算余弦相似度
import math
import httpx

OLLAMA_URL = "http://127.0.0.1:11434"
EMBED_MODEL = "bge-m3"

def get_embeddings(texts):
    """输入一个列表，返回对应的向量列表（批量，快）"""
    resp = httpx.post(
        OLLAMA_URL + "/api/embed",
        # 我传的是python字典 post方法按我提供的格式(如下) 将其转化成JSON
        # 并放进请求体再加上请求头 通过HTTP发给服务器
        json={"model": EMBED_MODEL, "input": texts},
        timeout=300,
    )
    resp.raise_for_status() # 检查状态码
    # 响应对象的 .json() 方法 把响应体的JSON字符串解析成 Python 字典
    # 从字典里取出 embeddings 这个键对应的值 把它返回
    return resp.json()["embeddings"]

def cosine(a, b):
    """余弦相似度 两个向量方向越一致 其夹角余弦值越接近 1"""
    # 公式 cos(θ) = (a · b) / (|a| × |b|) 点积/模积
    # zip(a, b) 把两个列表拉链式配对 (a[0], b[0]) (a[1], b[1])
    dot = sum(x * y for x, y in zip(a, b))
    # 算模长 每个元素的平方相加再开根号 math.aqrt() 开根号
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb)

if __name__ == "__main__": # 只有直接运行这个文件时 才会执行如下代码
    sentences = [
        "我喜欢小猫",
        "我养了一只猫",
        "今天天气不错",
        "汽车在高速公路上行驶",
    ]

    vecs = get_embeddings(sentences) # 把列表的每个语句向量化

    for i in range(len(sentences)):
        for j in range(i + 1, len(sentences)):
            score = cosine(vecs[i], vecs[j])
            print(sentences[i], "vs", sentences[j], "=", round(score, 4))
   
    print("向量维度：", len(vecs[0]))
