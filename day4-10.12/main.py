import sqlite3
import time
from datetime import datetime
import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import json
from fastapi.responses import StreamingResponse

app = FastAPI(title="My Agent API")

DB = "chat.db"
OLLAMA_URL = "http://127.0.0.1:11434"
MODEL = "qwen2.5-1.5b"
# 把提示词赋给一个变量
# SYSTEM_PROMPT = "你是一个乐于助人的中文助手，回答要简洁清晰。"
SYSTEM_PROMPT = "你是一名推理优化方向的资深工程师，回答专业且简洁"


# ---------- 数据库 ----------
def init_db():
    conn = sqlite3.connect(DB)
    # 多行数据库代码用 """    """
    conn.execute("""
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()


def save_message(session_id: str, role: str, content: str):
    conn = sqlite3.connect(DB)
    conn.execute(
        "INSERT INTO messages (session_id, role, content, created_at) VALUES (?, ?, ?, ?)",
        (session_id, role, content, datetime.now().isoformat(timespec="seconds")),
    )
    conn.commit()
    conn.close()


def get_history(session_id: str):
    conn = sqlite3.connect(DB)
    rows = conn.execute(
        "SELECT role, content, created_at FROM messages WHERE session_id = ? ORDER BY id",
        (session_id,),
    ).fetchall() # .fetchall() 把查到的所有行取出来 返回一个列表 列表的数据类型是元组
    conn.close()
    return rows # 返回元组类型的列表


def delete_history(session_id: str):
    conn = sqlite3.connect(DB)
    conn.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
    conn.commit()
    conn.close()


init_db()


# ---------- 数据模型 ----------
class ChatRequest(BaseModel):
    message: str
    session_id: str = "default"


class ChatResponse(BaseModel):
    reply: str
    session_id: str
    latency_ms: float # 这次回复花了多少毫秒


# ---------- 组装对话历史 ----------
def build_messages(session_id: str, max_messages: int = 10):
    """system 提示词 + 最近的历史消息（太多会变慢，所以限制条数）"""
    rows = get_history(session_id)
    # 创建字典类型的列表 放入 system 提示词
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for role, content, _ in rows[-max_messages:]:
        messages.append({"role": role, "content": content})
    return messages
"""
rows[-x] 取最后 x 条
rows[x] 取前 x 条
for a, b, c in row
循环变量的个数(这里也就是a, b, c) 必须与每个元素的长度相等(即 row 中的每个元素)
不需要的循环变量 用 _ 代替
"""


# ---------- 调用 Ollama ----------
async def ask_ollama(messages):
    async with httpx.AsyncClient(timeout=120) as client:
        resp = await client.post(
            OLLAMA_URL + "/api/chat",
            json={"model": MODEL, "messages": messages, "stream": False},
        )
        resp.raise_for_status()
        data = resp.json()
        return data["message"]["content"]
"""
异步上下文管理器 async with ... as client
进入时创建客户端 退出时自动关闭连接 释放资源
client 变量名 后面用它发请求
httpx 一个 HTTP 客户端库 能发网络请求 支持异步
AsyncClient 异步版客户端
timeout=120 超时时间 120 秒 响应时间在此之内不会报错
AsyncClient 是 httpx 库里的一个类 client 是这个类的一个实例化对象名
"""
"""
上下文管理器 with ... as ... 
自动借还 进入代码块时获取资源 离开代码块时自动释放资源 不必手动释放
"""
"""
resp = await client.post()
执行协程函数 要用 await
post() 是 AsyncClient 类的一个方法 提交数据给服务器处理
参数
url 是提交数据的服务器地址 + 具体接口
这里服务器地址就是 Ollama 本地大模型提供的固定地址
接口就是固定的对话接口 /api/chat
json 把字典转化成 JSON 字符串
把字典赋值给 json httpx 会将其自动转化成 JSON
字典里的 key 是 Ollama 官方文档规定的 必须按他的要求传
stream 控制怎么返回结果
stream=False 等模型把整段回答全生成完 一次性返回
stream=True 模型每生成一小块就立刻返回
返回值 Response 对象 其有多个属性
"""

async def stream_ollama(messages, session_id: str):
    """向 Ollama 发起流式请求，把增量文字一段段 yield 出去"""
    full_reply = []          # 收集完整回复，最后存数据库

    async with httpx.AsyncClient(timeout=120) as client:
        async with client.stream(
            "POST",
            OLLAMA_URL + "/api/chat",
            json={"model": MODEL, "messages": messages, "stream": True},
        ) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line:
                    continue # 跳过空行
                data = json.loads(line)
                piece = data.get("message", {}).get("content", "")
                if piece:
                    full_reply.append(piece)
                    yield "data:" + piece + "\n\n"          # 关键：每收到一段就立刻吐给前端

    # 流结束后，把完整回复存进数据库
    save_message(session_id, "assistant", "".join(full_reply))
"""
函数里出现了 yield 是生成器函数 调用它不会立刻执行 每次被迭代时执行到 yield 就暂停一下
httpx.AsyncClient(timeout=120) 异步版本的 HTTP 客户端 httpx 是一个 HTTP 客户端库
timeout=120 最多等 120 秒 超时就报错
async with ... as client 异步上下文管理器 代码结束时自动关闭客户端 不用手动清理 client 是实例化对象名
"""
"""
异步调用 client 对象的 stream 方法 把拿到的响应对象交给 resp
三个参数 POST HTTP方法 完整URL->服务器地址+具体接口 把传入的python对象自动转成json
"""
"""
resp.raise_for_status() 检查 HTTP 状态码 出错会抛出异常
"""
"""
async for 异步循环  aiter_lines() 收到一行就给一行 不用等全部收完
Ollama 流式返回的是 NDJSON 每行一个 JSON
data = json.loads(line) 把每一行 JSON 解析成字典
piece = data.get("message", {}).get("content", "")
先从 data 中取 message 字段 没有返回空字典 {}
再从 message 中取 content 没有就返回空字符
"""
"""
函数执行到 yield 处会暂停 把值交出去
下次被迭代时 从暂停处继续执行 这是流式的关键
"""

# ---------- 接口 ----------
@app.get("/")
def root():
    return {"message": "服务已启动", "model": MODEL}


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    if not req.message.strip():
        raise HTTPException(status_code=400, detail="message 不能为空")

    # 1. 存用户消息
    save_message(req.session_id, "user", req.message)

    # 2. 拼出给模型的消息（system + 历史）
    messages = build_messages(req.session_id)

    # 3. 调用模型（计时！）
    start = time.time()
    try:
        reply = await ask_ollama(messages)
    except httpx.HTTPError as e:
        raise HTTPException(status_code=502, detail="模型服务调用失败：" + str(e))
    latency_ms = round((time.time() - start) * 1000, 1)

    # 4. 存模型回复
    save_message(req.session_id, "assistant", reply)

    return ChatResponse(reply=reply, session_id=req.session_id, latency_ms=latency_ms)


@app.get("/history/{session_id}")
def history(session_id: str):
    rows = get_history(session_id)
    return {
        "session_id": session_id,
        "count": len(rows),
        "messages": [{"role": r[0], "content": r[1], "time": r[2]} for r in rows],
    }


@app.delete("/history/{session_id}")
def clear_history(session_id: str):
    delete_history(session_id)
    return {"message": "已清空", "session_id": session_id}


@app.get("/model")
async def list_models():
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.get(OLLAMA_URL + "/api/tags")
        resp.raise_for_status()
        data = resp.json()
    return {"models": [m["name"] for m in data.get("models", [])]}

@app.post("/chat/stream") # 流式接口
async def chat_stream(req: ChatRequest):
    if not req.message.strip(): # 为空抛出异常
        raise HTTPException(status_code=400, detail="message 不能为空")

    save_message(req.session_id, "user", req.message) # 调用函数 存对话信息到数据库
    messages = build_messages(req.session_id) # 拼装发给大模型的消息列表

    return StreamingResponse(
        stream_ollama(messages, req.session_id),
        media_type="text/event-stream; charset=utf-8",
    )
"""
StreamingResponse 是 FastAPI 的响应类 可以边产生数据边发送 而不是等所有内容生成完再一次性返回
stream_ollama(messages, req.session_id) 把生成器对象传给 StreamResponse
其自己去执行 每 yield 一段文字 就立刻发给客户端
media_type="text/plain; charset=utf-8" 响应头里的 Content-Type
text/plain
"""

# 作业 1：把流式接口改成标准 SSE 格式
#   每段输出改成 "data: " + piece + "\n\n"
#   media_type 改成 "text/event-stream"
#   （SSE 是浏览器 EventSource 的标准格式，前端对接更规范）

# 作业 2：用 test_stream.py 测 3 个不同长度的问题（短/中/长），
#   记录每次的 TTFT 和总耗时，填成表格
"""
长度------TTFT------总耗时
100------0.389------0.809
300------0.377------1.284
600------0.42------3.464
"""

# 作业 3（进阶思考）：流式输出对"Agent 调用工具"意味着什么？
#   提示：Agent 可能要先思考再调用工具再回答，
#   哪些部分可以流式给用户看（比如"正在检索…"）？
"""
在 agent 场景下 流式输出可以让用户看到 agent 在执行什么
可以流式给用户
正在思考…
正在检索知识库…（工具调用开始）
找到 5 个相关片段（工具调用结束）
最终回答的文字（逐字）
完成（结束事件）
"""    