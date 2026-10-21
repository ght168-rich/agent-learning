import json
import time
import httpx

start = time.time()
ttft = None
text = ""

with httpx.stream(
    "POST",
    "http://127.0.0.1:8000/ask/stream",
    json={"question": "什么是RAG？", "top_k": 3},
    timeout=300,
) as resp:
    for line in resp.iter_lines():
        if not line or not line.startswith("data: "):
            continue
        data = json.loads(line[6:])
        if "text" in data:
            if ttft is None:
                ttft = time.time() - start
            text += data["text"]
            print(data["text"], end="", flush=True)
        if data.get("done"):
            break

print(f"\n\nTTFT {ttft:.3f}s | 总耗时 {time.time() - start:.3f}s")