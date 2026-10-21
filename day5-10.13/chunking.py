# 把长文档切成(chunk)小文本块 重叠让每块都保留一点上下文 意思不会断
def split_text(text, chunk_size=200, overlap=50):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        c = text[start: end] # 1. 取一块 text[start:end]
        chunks.append(c) # 2. 加进 chunks
        start = end - overlap # 3. start 移到 end - overlap（制造重叠）
    return chunks

if __name__ == "__main__":
    with open("sample.txt", "r", encoding="utf-8") as f:
        text = f.read()
    chunks = split_text(text)
    print("总块数：", len(chunks))
    for i, c in enumerate(chunks):
        print(f"--- 第 {i} 块（{len(c)} 字）---")
        print(c[:60] + "...")
    # 检验重叠生效 下一块的开头和上一块的结尾应是同一段文字
    for i in range(len(chunks) - 1):
        print(f"第{i}块结尾: ...{chunks[i][-30:]}")
        print(f"第{i+1}块开头: {chunks[i+1][:30]}...")
        print()
