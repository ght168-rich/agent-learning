import json
import statistics
import time
import httpx

URL = "http://127.0.0.1:8000/ask"
payload_base = {"question": "RAG系统分为哪两个阶段？", "top_k": 3}

# 预热
httpx.post(URL, json={**payload_base, "alpha": 0.7}, timeout=300)

for alpha in (1.0, 0.7, 0.5):
    times = []
    for _ in range(5):
        p = {**payload_base, "alpha": alpha}
        t0 = time.time()
        r = httpx.post(URL, json=p, timeout=300).json()
        times.append((time.time() - t0) * 1000)

    print(
        f"alpha={alpha}: median={statistics.median(times):.1f}ms "
        f"p95={max(times):.1f}ms "
        f"retrieval={r['retrieval_ms']}ms generation={r['generation_ms']}ms"
    )
"""
实验记录
alpha------median(中位数)------p95(95分位延迟)------retrieval------generation
1.0-------489.1ms------523.4ms------114.2ms------222.7ms
0.7------474.1ms------483.1ms------116.7ms------231.8ms
0.5------466.7ms------480.6ms------124.7ms------236.4ms
"""