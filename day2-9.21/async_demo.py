import asyncio
import time
from fastapi import FastAPI

app = FastAPI()

@app.get("/slow/{seconds}")
async def slow(seconds: int):
    # 把控制权交给调度器 seconds 秒 调度器可以进行其他工作 时间结束后交回控制权
    await asyncio.sleep(seconds)
    return {"slept": seconds}

@app.get("/demo/concurrent")
async def concurrent():
    start = time.time()

    # 并发执行 3 个耗时任务（注意：是同时跑，不是一个个排队）
    results = await asyncio.gather(slow(1), slow(1), slow(1))

    elapsed = round(time.time() - start, 2)
    return {"elapsed": elapsed, "results": results}
"""
协程
可以中途暂停 之后从暂停处继续执行的函数
async def 定义的函数叫协程函数 直接调用它不会执行 只会返回一个协程对象
执行协程函数 await 函数名() 或 asyncio.run(函数名())1
"""
"""
串行
对协程函数逐一 await 执行
并发
用 await asyncio.gather() 参数是要并发执行的多个协程函数
"""