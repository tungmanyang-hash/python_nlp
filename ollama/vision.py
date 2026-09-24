import asyncio
from ollama import AsyncClient
import time

image_path = './example01.jpg'

OLLAMA_HOST = "http://localhost:11434"

messages = [
    {
    'role': 'user',
    'content': '請描述這張圖片的內容。',
    'images': {image_path}
    }
]

async def recognize():
    t1 = time.time()

    client = AsyncClient(
        host=OLLAMA_HOST,
        timeout=600
    )

    response = await client.chat(
        model='gemma4:e2b',
        messages=messages,
        keep_alive="1h",
        think=False,
        options={
                    "temperature": 1.0,
                    "top_k": 64,
                    "top_p": 0.95
                },
    )
    print(response.message.content)

    t2 = time.time()
    print(f"Response time: {t2 - t1:.2f} seconds")

if __name__ == '__main__':
    asyncio.run(recognize())