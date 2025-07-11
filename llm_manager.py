import json
import chromadb
from sentence_transformers import SentenceTransformer
from llm_embedder import LLMEmbedder
from llm_knowledge_base import KnowledgeBase

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
client = Groq(api_key=os.getenv("GROQ_KEY"))
# ----------------

class LLMManager:
    def __init__(self):
        self.embedder = LLMEmbedder(config['embedding_model']['name'], client)
        self.knowledge_base = KnowledgeBase(self.embedder, collection)

    def embed(self, text):
        return self.embedder.embed(text)
      
    def query(self, query):
        """
        Process the user query to extract relevant information and retrieve context.
        """
        context_array = self.knowledge_base.query(query)
        
        if not context_array:
            return "Not enough information in the context to answer this question."
        
        return self._query(query, context_array)

    def _query(self, query, context_array):
        """
        Process the user query and context array to generate a response.
        """
        context_block = "\n---\n".join(context_array)
        system_prompt = f"""
        You are an expert on the video game "Enter the Gungeon". Use the context below to answer the user question. Do not make up information not found in the context. Be as concise as you can while still providing a complete answer. If the context does not contain enough information to answer the question, say "I don't know" or "Not enough information in the context to answer this question.".
        Context:
        {context_block}
        
        User Question:
        {query}
        """
        
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query}
            ],
            max_tokens=500,
            temperature=0.0
        )
        if response.choices:
            return response.choices[0].message.content.strip()