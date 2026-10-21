import time
import httpx

start = time.time()
first_token_time = None

with httpx.stream(
    "POST",
    "http://127.0.0.1:8000/chat/stream",
    json={"message": "用600字解释什么是大模型推理", "session_id": "stream_test"},
    timeout=120,
) as resp:
    for chunk in resp.iter_text():
        if chunk:
            if first_token_time is None:
                first_token_time = time.time() - start
                print("\n>>> TTFT(首字延迟):", round(first_token_time, 3), "秒\n")
            print(chunk, end="", flush=True)

total = time.time() - start
print("\n\n>>> 总耗时:", round(total, 3), "秒")
