import os
import sqlite3
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

model_name = 'BAAI/bge-m3'
index_path = './vector.index'
encoder = SentenceTransformer(model_name)
docs = []
doc_ids = []
conn = sqlite3.connect('./news.db')
cursor = conn.cursor()

try:
    rows = cursor.execute(
        'SELECT id, title, content FROM news'
    )

    for news_id, title, content in rows:
        docs.append(f"{title} {content}")
        doc_ids.append(news_id)
finally:
    conn.close()

doc_ids = np.array(doc_ids, dtype='int64')

embeddings = encoder.encode(
    docs,
    batch_size=8,
    show_progress_bar=True,
    normalize_embeddings=True
)

index = None

if not os.path.exists(index_path):
    dims = embeddings.shape[1]
    index = faiss.IndexFlatIP(dims)
    index  = faiss.IndexIDMap(index)
else:
    index = faiss.read_index(index_path)

print(embeddings.shape)
print(len(doc_ids))
index.add_with_ids(embeddings, doc_ids)
faiss.write_index(index, index_path)
print('索引已儲存到' , index_path)