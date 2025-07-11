import json
from llm_embedder import LLMEmbedder
from llm_knowledge_base import KnowledgeBase
from persistent_chat import ChatHistory

# --- SETUP ---
with open("config.json", "r") as f:
    config = json.load(f)


import os
import dotenv
dotenv.load_dotenv()

from groq import Groq
client = Groq(api_key=os.getenv("GROQ_KEY"))
# ----------------

class LLMManager:
    def __init__(self, config, persistent=False):
        self.config = config
        self.embedder = LLMEmbedder(client, config)
        self.knowledge_base = KnowledgeBase(self.embedder, config)
        self.persistent = persistent
        if self.persistent:
            self.chat_history = ChatHistory(
                chat_limit=config['chat_history']['chat_limit'],
                context_limit=config['chat_history']['context_limit']
            )

    def embed(self, text):
        return self.embedder.embed(text)
      
    def query(self, query):
        """
        Process the user query to extract relevant information and retrieve context, and combine it with previous context
        """
        
        additional_context = ""
        if self.persistent:
            additional_context = str(self.chat_history)
        context_array = self.knowledge_base.query(query)
        
        if not context_array:
            return "Not enough information in the context to answer this question."
        
        answer = self._query(query, context_array, additional_context)
        if self.persistent:
            self.chat_history.inqueue_context("\n---\n".join(context_array))
            self.chat_history.inqueue_message("user", query)
            self.chat_history.inqueue_message("assistant", answer)
        return answer

    def _query(self, query, context_array, additional_context=None):
        """
        Process the user query and context array to generate a response.
        """
        context_block = "\n---\n".join(context_array)
        system_prompt = f"""
        You are an expert on the video game "Enter the Gungeon". Use the context below to answer the user question. Do not make up information not found in the context. Be as concise as you can while still providing a complete answer. {"Previous Context and conversations is included. " if len(additional_context) else ""}If the context does not contain enough information to answer the question, say "I don't know" or "Not enough information in the context to answer this question.".

        {additional_context if len(additional_context) else ""}

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