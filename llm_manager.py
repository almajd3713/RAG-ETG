import json
from llm_embedder import LLMEmbedder
from llm_knowledge_base import KnowledgeBase
from chat_history import ChatHistory
import logging
import datetime

# Set up logging

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
    def __init__(self, config, persistent=False, log_dir=None):
        self.config = config
        if log_dir:
            log_dir = os.path.join(log_dir, "llm_manager")
            self.logger = True
            if not os.path.exists(log_dir):
              os.makedirs(log_dir)
            current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            logging.basicConfig(level=logging.INFO, filename=os.path.join(log_dir, f"{current_time}.log"),
                                format='%(asctime)s - %(levelname)s - %(message)s')
        
        self.persistent = persistent
        if self.persistent:
            self.chat_history = ChatHistory(
                chat_limit=config['chat_history']['chat_limit'],
                context_limit=config['chat_history']['context_limit']
            )
        else: self.chat_history = None
        self.embedder = LLMEmbedder(client, config, logging)
        self.knowledge_base = KnowledgeBase(self.embedder, self.chat_history, config, logging)


    def embed(self, text):
        return self.embedder.embed(text)
      
    def query(self, query):
        """
        Process the user query to extract relevant information and retrieve context, and combine it with previous context
        """
        try:
            additional_context = None
            if self.persistent:
                additional_context = str(self.chat_history)
            reformulated_query, context_array = self.knowledge_base.query(query)
            
            answer = self._query(reformulated_query, context_array=context_array, additional_context=additional_context)
            if self.persistent:
                if context_array: self.chat_history.inqueue_context(context_array)
                self.chat_history.inqueue_message("user", query)
                self.chat_history.inqueue_message("assistant", answer)
            return answer
        except Exception as e:
            logging.error(f"Error in LLMManager query: {e}")
            return "An error occurred while processing your query. Please try again later."

    def _query(self, query, context_array=None, additional_context=None):
        """
        Process the user query and context array to generate a response.
        """
        context_block = "\n".join([f"\n--- Document ---\n{c}\n--- Document End ---" for c in context_array]) if context_array else None
        system_prompt = f"""
        You are an expert on the video game "Enter the Gungeon". Use the context below to answer the user question. Do not make up information not found in the context. Be as concise as you can while still providing a complete answer. {"Previous Context and conversations is included. " if additional_context else ""}If the context does not contain enough information to answer the question, say "I don't know" or "Not enough information in the context to answer this question.".

        {additional_context if additional_context else ""}

        {f"Context: {context_block}" if context_block else ""}
        """
        logging.info(f"System Prompt: {system_prompt}")
        logging.info(f"User Query: {query}")
        
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
            result = response.choices[0].message.content.strip()
            if self.logger: logging.info(f"LLM Response: {result}")
            return result