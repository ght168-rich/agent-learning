import sqlite3
import time
from datetime import datetime
import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

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
    