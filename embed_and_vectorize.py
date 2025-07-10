import chromadb
from sentence_transformers import SentenceTransformer

client = chromadb.PersistentClient(
    path="chroma_db",
    settings=chromadb.Settings(allow_reset=True)
)

import json
import logging
import datetime
import os

with open("config.json", "r") as f:
    config = json.load(f)

log_dir = "logs/embed_and_vectorize"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
log_file = os.path.join(log_dir, f"{current_time}.log")
logging.basicConfig(filename=log_file, level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

def load_data(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data

def embed_and_vectorize_data(data, model_name, model):
  collection_name = f"rag_etg_{model_name}"
  collection = client.create_collection(name=collection_name)
  for idx, chunk in enumerate(data, 1):
    text_to_embed = chunk["text"]
    emb = model.encode(
      text_to_embed, normalize_embeddings=True,
      device='cuda', batch_size=64, show_progress_bar=False
    )
    collection.add(
      documents=[text_to_embed],
      metadatas=[{**chunk["meta"], "id": chunk["id"]}],
      ids=[chunk["id"]],
      embeddings=[emb.tolist()]
    )
    if idx % 100 == 0:
      print(f"Processed {idx} chunks")
      logging.info(f"Processed {idx} chunks for model {model_name}")

def save_vectors():
  client.persist()
  
  
if __name__ == "__main__":
    client.reset()  # Reset the client to start fresh
    logging.info("Loading data from all_chunks.json...")
    data = load_data("all_chunks.json")
    logging.info(f"Loaded {len(data)} chunks.")

    logging.info("Loading embedding models...")
    # models = {
        # "bge": SentenceTransformer("BAAI/bge-base-en-v1.5"),
        # "e5": SentenceTransformer("intfloat/e5-base-v2"),
        # "minillm": SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    # }
    logging.info("Models loaded successfully.")

    logging.info("Starting embedding and vectorization process...")
    model, model_name = config['embedding_model']['name'], config['embedding_model']['collection_name']
    model = SentenceTransformer(model)
    logging.info(f"Embedding and vectorizing using {model_name} model...")
    embed_and_vectorize_data(data, model_name, model)
    logging.info(f"Data embedded, vectorized, and stored in collection: rag_etg_{model_name}")
    logging.info("Embedding and vectorization process completed.")