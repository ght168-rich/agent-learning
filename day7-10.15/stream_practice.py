import json
import time
import httpx

OLLAMA_URL = "http://127.0.0.1:11434"
CHAT_MODEL = "qwen2.5-1.5b"


def ask_ollama(prompt):
    start = time.time()
    resp = httpx.post(
        OLLAMA_URL + "/api/chat",
        json={"model": CHAT_MODEL,
              "messages": [{"role": "user", "content": prompt}],
              "stream": False,
              "options": {"temperature": 0}},
        timeout=300,
    )
    resp.raise_for_status()
    return resp.json()["message"]["content"], time.time() - start


def ask_ollama_stream(prompt):
    full = []
    start = time.time()
    ttft = None

    # 记住参数怎么写
    with httpx.stream(
        "POST",
        OLLAMA_URL + "/api/chat",
        json={
            "model": CHAT_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "stream": True,
            "options": {"temperature": 0}
        },
        timeout=300,
    ) as resp:
        resp.raise_for_status()
        for line in resp.iter_lines():
            if not line:
                continue
            data = json.loads(line)
            piece = data.get("message", {}).get("content", "")
            if piece:
                if ttft is None:
                    ttft = time.time() - start
                full.append(piece)
                print(piece, end="", flush=True)

    total = time.time() - start
    return "".join(full), ttft, total


if __name__ == "__main__":
    prompt = "用150字解释什么是大模型推理的prefill和decode阶段。"

    print("--- 非流式 ---")
    ans1, t1 = ask_ollama(prompt)
    print(ans1)
    print(f"总耗时 {t1:.2f}s（TTFT = 总耗时）")

    print("\n--- 流式 ---\n")
    ans2, ttft, t2 = ask_ollama_stream(prompt)
    print(f"\n\nTTFT {ttft:.3f}s | 总耗时 {t2:.3f}s")