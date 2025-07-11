
# --- SETUP ---

# ----------------

class ChatHistory:
    def __init__(self, chat_limit=12, context_limit=6, logger=None):
        self.message_history = []
        self.context_history = []
        self.chat_limit = chat_limit
        self.context_limit = context_limit
        self.logger = logger

    def inqueue_message(self, role, message):
        if self.logger:
            self.logger.info(f"Inqueue message: {role}: {message}")
        if isinstance(message, str):
            message = {"role": role, "text": message}
        self.message_history.append(message)
        if len(self.message_history) > self.chat_limit:
            self.dequeue_message()

    def dequeue_message(self):
        if self.logger:
            self.logger.info(f"Dequeue message: {self.message_history[0] if self.message_history else 'None'}")
        if self.message_history:
            return self.message_history.pop(0)
        return None

    def inqueue_context(self, context):
        if self.logger:
            self.logger.info(f"Inqueue context: {context[:100]}...")
        self.context_history.append(context)
        if len(self.context_history) > self.context_limit:
            self.dequeue_context()

    def dequeue_context(self):
        if self.logger:
            self.logger.info(f"Dequeue context: {self.context_history[0][:100] if self.context_history else 'None'}...")
        if self.context_history:
            return self.context_history.pop(0)
        return None

    def __str__(self):
      return """
      Previous Context:
      """ + "\n".join(self.context_history) + """
      Previous Messages:
      """ + "\n".join([f"{m['role']}: {m['text']}" for m in self.message_history]) + """
    """
    
    def get_chat(self):
        return f'{"\n".join([f"{m["role"]}: {m["text"]}" for m in self.message_history])}'

    def get_context(self):
        return "\n".join(self.context_history)