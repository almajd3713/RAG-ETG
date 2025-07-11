import json
import chromadb
from sentence_transformers import SentenceTransformer

# --- SETUP ---
with open("config.json", "r") as f:
    config = json.load(f)
collection_name = f"rag_etg_{config['embedding_model']['collection_name']}"
client = chromadb.PersistentClient(
    path="chroma_db",
)
collection = client.get_collection(
    name=collection_name
)
model = SentenceTransformer(config['embedding_model']['name'])

import os
import dotenv
dotenv.load_dotenv()
from groq import Groq
# Initialize Groq client
client = Groq(api_key=os.getenv("GROQ_KEY"))
# ----------------

class ChatHistory:
    def __init__(self, chat_limit=12, context_limit=6):
        self.message_history = []
        self.context_history = []
        self.chat_limit = chat_limit
        self.context_limit = context_limit

    def inqueue_message(self, role, message):
        if isinstance(message, str):
            message = {"role": role, "text": message}
        self.message_history.append(message)
        if len(self.message_history) > self.chat_limit:
            self.dequeue_message()

    def dequeue_message(self):
        if self.message_history:
            return self.message_history.pop(0)
        return None

    def inqueue_context(self, context):
        if isinstance(context, str):
            context = {"text": context}
        self.context_history.append(context)
        if len(self.context_history) > self.context_limit:
            self.dequeue_context()

    def dequeue_context(self):
        if self.context_history:
            return self.context_history.pop(0)
        return None

    def __str__(self):
      return """
      Previous Context:
      """ + "\n".join([c['text'] for c in self.context_history]) + """
      Previous Messages:
      """ + "\n".join([f"{m['role']}: {m['text']}" for m in self.message_history]) + """
    """
