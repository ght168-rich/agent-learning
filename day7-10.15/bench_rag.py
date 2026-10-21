import json
import time
import httpx

ASK = "http://127.0.0.1:8000/ask"
STREAM = "http://127.0.0.1:8000/ask/stream"
payload = {"question": "什么是RAG？", "top_k": 3}

# 预热：让模型加载 + 缓存生效（这轮不算）
print("预热中...")
httpx.post(ASK, json=payload, timeout=300)

print("\n轮次 | 非流式(总) | 流式TTFT | 流式(总)")
for i in range(1, 4):
    t0 = time.time()
    r = httpx.post(ASK, json=payload, timeout=300).json()
    t_total = time.time() - t0

    start = time.time()
    ttft = None
    with httpx.stream("POST", STREAM, json=payload, timeout=300) as resp:
        for line in resp.iter_lines():
            if not line.startswith("data: "):
                continue
            d = json.loads(line[6:])
            if "text" in d and ttft is None:
                ttft = time.time() - start
            if d.get("done"):
                break
    t2 = time.time() - start

    print(f"{i}    | {t_total:.2f}s      | {ttft:.3f}s   | {t2:.2f}s")

print("\n最近一次 /ask 内部耗时：")
print("检索:", r["retrieval_ms"], "ms | 生成:", r["generation_ms"], "ms")