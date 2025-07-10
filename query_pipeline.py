import chromadb

collection_name = "rag_etg_minillm"
client = chromadb.PersistentClient(
    path="chroma_db",
)
collection = client.get_collection(
    name=collection_name
)

from sentence_transformers import SentenceTransformer
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

query = "Give me trivia information about the 00"
emb = model.encode(query, normalize_embeddings=True)

results = collection.query(
    query_embeddings=[emb],
    n_results=5
)

print("Query Results:")
for idx, (doc, meta, dist) in enumerate(zip(results['documents'][0], results['metadatas'][0], results['distances'][0])):
    print(f"\n[Result {idx + 1}]")
    print(f"Title   : {meta['title']}")
    print(f"Section : {meta['section']}")
    print(f"ID      : {meta['id']}")
    print(f"Distance: {dist:.3f}")
    print("Text    :")
    print(doc)