import sqlite3
from datetime import datetime
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="My Agent API")

# chat.db 是数据库文件的名字 DB str 类型变量 存着文件名
DB = "chat.db"

# ---------- 数据库相关 ----------
def init_db():
    """启动时建表（如果表不存在就创建）"""
    # 打开或创建一个数据库文件 参数是文件名 返回一个 conn 对象 之后操作都要通过它
    conn = sqlite3.connect(DB)
    # 数据库对象名.execute() 执行SQL语句 
    conn.execute("""
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
    """)
    conn.commit() # 把改动写入文件
    conn.close() # 关闭连接 释放资源

def save_message(session_id: str, role: str, content: str):
    """存一条消息（role 是 user 或 assistant）"""
    conn = sqlite3.connect(DB)
    conn.execute(
        "INSERT INTO messages (session_id, role, content, created_at) VALUES (?, ?, ?, ?)",
        (session_id, role, content, datetime.now().isoformat(timespec="seconds")),
    )
    # 参数化查询 conn.execute("INSERT INTO messages (...) VALUES (?, ?, ?, ?)", (a, b, c, d))
    conn.commit()
    conn.close()

def get_history(session_id: str):
    """按时间顺序取出某个会话的所有消息"""
    conn = sqlite3.connect(DB)
    rows = conn.execute(
        "SELECT role, content, created_at FROM messages WHERE session_id = ? ORDER BY id",
        (session_id,),
    ).fetchall()
    """
    OEDER BY id 按 id 从小到大排
    .fetcchall() 把查到的所有行取出来 返回一个列表 列表的数据类型是元组
    """
    conn.close()
    return rows

def delete_history(session_id: str):
    conn = sqlite3.connect(DB)
    conn.execute(
        # 只有一个参数时一定要加 , 这才会识别成元组
        "DELETE FROM messages WHERE session_id = ?", (session_id,)
    )
    conn.commit()
    conn.close()

init_db() # 执行创建表函数 程序启动时执行一次

# ---------- 数据模型 ----------
class ChatRequest(BaseModel):
    message: str
    session_id: str = "default"


class ChatResponse(BaseModel):
    reply: str
    session_id: str

# ---------- 接口 ----------
@app.get("/")
def root():
    return {"message": "服务已启动"}

@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    # 输入校验：空消息直接拒绝（400 = 客户端请求有问题）
    # .strip() 去掉首尾空格
    if not req.message.strip():
        raise HTTPException(status_code=400, detail="message 不能为空")
    if len(req.message) > 20:
        raise HTTPException(status_code=400, detail="messages 长度不能超过20")
    """
    raise 抛出异常
    HTTPException() 异常类 内置两个参数
    status_code= 状态码
    detail= 错误说明
    """

    # 1. 存用户说的话
    save_message(req.session_id, "user", req.message)

    # 2. 生成回复（现在是假的，明天接 Ollama）
    reply = "你说的是：" + req.message + "（假回复，明天接入真模型）"

    # 3. 存模型的回复
    save_message(req.session_id, "assistant", reply)

    return ChatResponse(reply=reply, session_id=req.session_id)

@app.get("/history/{session_id}")
def history(session_id: str):
    """查看某个会话的全部历史"""
    rows = get_history(session_id)
    return {
        "session_id": session_id,
        "count": len(rows),
        # 元组按下标取值 从 0 开始
        "messages": [
            # 对 rows 里的每一个 r 都生成字典 放进列表里
            {"role": r[0], "content": r[1], "time": r[2]} for r in rows
        ]
    }

# 作业 1：加接口 DELETE /history/{session_id}，清空该会话的历史
#         提示：conn.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
@app.delete("/history/{session_id}")
def clear_history(session_id: str):
    delete_history(session_id)
    return {"message": "已清空", "session_id": session_id}
    
# 作业 2：给 /chat 加校验：message 长度超过 500 就返回 400

# 作业 3：加接口 GET /sessions，返回所有出现过的 session_id（去重）
#         提示：SELECT DISTINCT session_id FROM messages
@app.get("/session")
def show():
    conn = sqlite3.connect(DB)
    result = conn.execute(
        "SELECT DISTINCT session_id FROM messages"
    ).fetchall()
    return [
        "session_id=" + r[0] for r in result
    ]
