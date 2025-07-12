import json
from llm.llm_embedder import LLMEmbedder
from llm.llm_knowledge_base import KnowledgeBase
from llm.llm_engines import get_engine
from chat.chat_history import ChatHistory
import logging
import datetime
import os

# --- SETUP ---
with open("config.json", "r") as f:
    config = json.load(f)
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
        self.engine = get_engine(
            config['llm_engine']['platform'], 
            config['llm_engine']['model_name'], 
            config['llm_engine']['defaults']
        )
        self.embedder = LLMEmbedder(self.engine, config, logging)
        self.knowledge_base = KnowledgeBase(self.embedder, self.chat_history, config, logging)


    def embed(self, text):
        return self.embedder.embed(text)
      
    def query(self, query):
        """
        Process the user query to extract relevant information and retrieve context, and combine it with previous context
        """
        try:            
            # Although I'd rather add all three at the same time, focus does consider the current query in its decision
            if self.persistent:
                self.chat_history.inqueue_message("user", query)
            conversation_focus = self._get_conversation_focus()
            
            additional_context = None
            if self.persistent:
                additional_context = str(self.chat_history)
            query_info, context_array = self.knowledge_base.query(query, conversation_focus)

            answer = self._query(query_info['query'], 
                context_array=context_array, 
                additional_context=additional_context,
                conversation_focus=conversation_focus
            )
            # If the LLM response is not satisfactory and the query was believed to have enough context, request additional context.
            if answer in ["I don't know", "Not enough information in the context to answer this question."] and not context_array:
                if self.logger: logging.info("Although the query was believed to have enough context, the LLM could not answer it. Requesting additional context via lookup.")
                context_array = self.knowledge_base._query_forced(query, conversation_focus)
                answer = self._query(query_info['query'], 
                    context_array=context_array, 
                    additional_context=additional_context,
                    conversation_focus=conversation_focus
                )
                
            if self.persistent:
                if context_array: self.chat_history.inqueue_context(context_array)
                self.chat_history.inqueue_message("assistant", answer)
            return answer
        except Exception as e:
            logging.error(f"Error in LLMManager query: {e}")
            return "An error occurred while processing your query. Please try again later."

    def _query(self, query, context_array=None, additional_context=None, conversation_focus=None):
        """
        Process the user query and context array to generate a response.
        """
        context_block = "\n".join([f"\n--- Document ---\n{c}\n--- Document End ---" for c in context_array]) if context_array else None
        system_prompt = f"""
        You are an expert on the video game "Enter the Gungeon". Use the context below to answer the user question. Do not make up information not found in the context. Be as concise as you can while still providing a complete answer. {"Previous Context and conversations is included. " if additional_context else ""}If the context does not contain enough information to answer the question, say "I don't know" or "Not enough information in the context to answer this question.".

        {additional_context if additional_context else ""}

        {f"Context: {context_block}" if context_block else ""}

        {conversation_focus if conversation_focus else ""}
        """
        logging.info(f"System Prompt: {system_prompt}")
        logging.info(f"User Query: {query}")
        
        response = self.engine.generate_response({
            "system_query": system_prompt,
            "user_query": query,
            "max_tokens": self.config['llm_engine']['defaults'].get('max_tokens', 500),
            "temperature": self.config['llm_engine']['defaults'].get('temperature', 0.0)
        })
        if self.logger: logging.info(f"LLM Response: {response}")
        return response

    def _get_conversation_focus(self):
        """
        Get the conversation focus based on the chat history.
        """
        if not self.persistent or not len(self.chat_history.message_history):
            return None
        conversation = '\n'.join([f"{m['role']}: {m['text']}" for m in self.chat_history.message_history])
        if self.logger: logging.info(f"Current Conversation: {conversation}")
        
        system_prompt = f"""
        You are an expert on the video game "Enter the Gungeon". Based on the conversation below, Determine the main object of focus for the conversation, which may be referred to as "it" or "that". If there is no clear focus, return None.
        """
        user_query = f"Conversation: {conversation}\nWhat is the main object of focus for the conversation?"
        if self.logger: logging.info(f"System Prompt: {system_prompt}")
        if self.logger: logging.info(f"User Query: {user_query}")
        response = self.engine.generate_response({
            "system_query": system_prompt,
            "user_query": user_query,
            "max_tokens": 100,
            "temperature": 0.0
        })
        if response:
            if self.logger: logging.info(f"Conversation Focus: {response}")
            return response
        return None