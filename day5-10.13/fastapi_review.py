from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()                              # ①

class ChatRequest(BaseModel):
    message: str                       # ②
    session_id: str = "default"

@app.post("/chat")                      # ③
async def chat(req: ChatRequest):               # ④ ⑤
    return {"reply": "收到：" + req.message,   # ⑥
            "session_id": req.session_id}         # ⑦