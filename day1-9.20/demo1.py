from fastapi import FastAPI

app = FastAPI(title="My First Agent API")

@app.get("/")
def root():
    return {"message": "服务已启动"}


@app.get("/health")
def health():
    return {"status": "ok"}
"""
1.理解三个概念
@app.get("/") 装饰器 表示访问 / 网址时 执行下面函数 get 是获取数据
def root(): 被访问时执行的函数
return {...} 返回值会自动变成 JSON
"""
"""
2.运行
uvicorn demo1:app --reload
uvicorn 负责接收浏览器/客户端的 HTTP 请求 然后转发给 python 代码
demo1 python 文件名
:app 文件里的 FastAPI 对象名
--reload 改代码后 服务器自动重启
"""
"""
3.app = FastAPI(title="...")
创建一个 FastAPI 应用实例 title 是给 /docs 文档页面用的标题
app 是实例对象的名字 调用都要用到
装饰器 作用是给下面的函数贴一个标签
"""
