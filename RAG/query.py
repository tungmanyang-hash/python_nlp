from sentence_transformers import SentenceTransformer
import faiss
import sqlite3

model_name = 'BAAI/bge-m3'
encoder = SentenceTransformer(model_name)
index_path = './vector.index'
index = faiss.read_index(index_path)

list_query = ['有人用AI來偵測大腸息肉嗎?']
              
embeddings = encoder.encode(
    list_query,
    batch_size=3,
    show_progress_bar=False,
    normalize_embeddings=True
)      

D, I = index.search(embeddings, k=3)
list_scores = D.tolist()
list_ids = I.tolist()
print(f'相似度: {list_scores}')
print(f"檢索的 Document IDs 為: {list_ids}")

conn = sqlite3.connect('./news.db')

try:
    user_prompt = ''

    for query, id, scores in zip(list_query, list_ids, list_scores):
        user_prompt = '=' * 80
        user_prompt += f'\n使用者查詢的問題: {query}'
        
        for doc_id, scores in zip(ids, scores):
            user_prompt += f'\n{'-' * 80}'
            user_prompt += f'\nDocument ID:{doc_id}'
            user_prompt += f"\n相似度: {score}"

            row = conn.execute(
                    'SELECT title, content, summary FROM news WHERE id = ?',
                    (int(doc_id),)
                    ).fetchone()

            title, content, summary = row
            user_prompt += f"\n標題: {title}"
            user_prompt += f"\n文件摘要: {summary}" 

finally:
    conn.close()