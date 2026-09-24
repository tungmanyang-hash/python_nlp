import asyncio
from ollama import AsyncClient
import time
import math

OLLAMA_HOST = "http://localhost:11434"

def cosine_similarity(vector1, vector2):
    dot_product = 0
    length1 = 0
    length2 = 0

    for a, b in zip(vector1, vector2):
        dot_product += a * b
        length1 += a * a
        length2 += b * b

    length1 = math.sqrt(length1)
    length2 = math.sqrt(length2)

    if length1 == 0 or length2 == 0:
        return 0

    return dot_product / (length1 * length2)

async def compare_texts():
    text1 = '早安'
    text2 = '早上好'

    t1 = time.time()

    client =  AsyncClient(
        host=OLLAMA_HOST,
        timeout=600,
    )

    response = await client.embed(
        model='bge-m3',
        input=[text1, text2],
        keep_alive='1h',
    )

    embeddings = response.embeddings
    vector1 = embeddings[0]
    vector2 = embeddings[1]

    similarity = cosine_similarity(vector1, vector2)

    print(f"語意相似度: {similarity:.4f}")

    t2 = time.time()
    print(f"執行時間{t2 - t1:.2f}秒")

if __name__ == '__main__':
        asyncio.run(compare_texts())