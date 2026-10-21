from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="My First Agent API")

class ChatRequest(BaseModel):
    """请求的数据长什么样：一条消息 + 会话 ID"""
    message: str
    session_id: str = "default" # 不传默认是 default

class ChatResponse(BaseModel):
    """返回的数据长什么样"""
    reply: str
    session_id: str

@app.get("/")
def root():
    return {"message": "服务已启动"}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    # 现在是假回复，第 3-4 天会换成真的 Ollama 模型
    reply = "你说的是：" + req.message + "（这是假回复，过两天接入真模型）"
    return ChatResponse(reply=reply, session_id=req.session_id)
"""
1.class ChatRequest(BaseModel) class ChatResponse(BaseModel)
分别描述请求和返回的数据格式 都继承 BaseModel
Pydantic 是一个库 提供 BaseModel 类
Pydantic 带自动校验 传错误格式会返回 422 错误
"""
"""
2.@app.post("/chat", response_model=ChatResponse)
post 用 POST 方法提交数据
response_model=ChatResponse 告诉 FastAPI 这个接口返回的数据要符合 ChatResponse 的格式
它会自动写进 /docs 文案
"""
"""
3.async def chat(req: ChatRequest)
参数 req: ChatRequest req 是 ChatRequest 类的实例化对象
可以用 req.message req.session_id 取值
由于ChatRequest 属于 Pydantic 库 FastAPI 就会自动把请求体里的 JSON 转成 ChatRequest 对象传给你
"""
"""
4.
前端->后端 整体叫请求(request) 里面的数据叫 请求体(request body) 附加信息叫请求头(request headers)
后端->前端 整体叫响应(response) 里面的数据叫 响应体(response body) 附加信息叫响应头(reponse headers)
客户端发请求
POST /chat HTTP/1.1 第一行 方法 + 路径 + 协议版本
Content-Type: application/json 第二行 请求头(附加说明) 如 用JSON格式
                                空行
 {                             请求体 真正要传的数据
 "message": "你好",
  "session_id": "test1"
 }
 后端返回响应
HTTP/1.1 200 OK                  第一行 状态行 200 表示成功 + 协议版本
Content-Type: application/json   第二行 响应头 附加信息
                                 空行
{"reply": "你好呀"}               响应体 真正要传的数据                             
"""
"""
ChatResponse(...)
构造并返回给 FastAPI 一个 ChatResponse 类的实例化对象
FastAPI 收到后会转成 JSON 返回
"""
"""
async def 表示异步函数
同步(同步阻塞) 代码按顺序执行 遇到耗时操作就卡在那里等待其完成 再往下走
异步(异步阻塞) 遇到耗时操作不干等 把控制权交出去 先去处理别的任务 等待结束后再回来继续
正常的函数声明语法就是同步函数
async def 声明异步函数
异步函数里 等 要用 await 如 await asyncio.sleep(2) 等 2 秒 但这 2 秒里可以处理别的请求
要用 await 必须声明异步函数
"""
# 作业 1：GET /hello/{name}
# 访问 /hello/小明 时返回 {"hello": "小明"}（提示：函数参数写 def hello(name: str)）
@app.get("/hello/{name}")
def hello(name: str):
    return {"hello": name}

# 作业 2：POST /sum
# 请求 {"a": 3, "b": 5}，返回 {"result": 8}
# （提示：先定义 class SumRequest(BaseModel): a: int; b: int）
class SumRequest(BaseModel):
    a: int
    b: int

@app.post("/sum")
def add(req: SumRequest):
    return {"result": req.a + req.b}
# 作业 3：GET /slow
# 用 async def + await asyncio.sleep(2)，两秒后返回 {"done": True}
# （提示：文件顶部 import asyncio，函数里 await asyncio.sleep(2)）
import asyncio

@app.get("/slow")
async def slow():
    await asyncio.sleep(2)
    return {"done": True}
