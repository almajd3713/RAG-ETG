# ----- SETUP -----
from abc import abstractmethod
from groq import Groq
import os
import dotenv

dotenv.load_dotenv()

from google import genai
from google.genai import types
# -----------------

class LLMEngine:
    def __init__(self, model_name: str, defaults=None):
        self.model_name = model_name
        self.temperature = defaults.get("temperature", 0.0) if defaults else 0.0
        self.max_tokens = defaults.get("max_tokens", 500) if defaults else 500

    @abstractmethod
    def generate_response(self, params: dict) -> str:
        # Implementation for generating a response from the LLM
        pass

class GroqLLMEngine(LLMEngine):
    def __init__(self, model_name: str, defaults=None):
        """
        Initialize the Groq LLM engine with the model name and default parameters.
        """
        super().__init__(model_name, defaults)
        self.groq_client = Groq(os.getenv("GROQ_KEY"))

    def generate_response(self, params: dict) -> str:
        response = self.groq_client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": params.get("system_query", "")},
                {"role": "user", "content": params.get("user_query", "")}
            ],
            max_tokens=params.get("max_tokens", self.max_tokens),
            temperature=params.get("temperature", self.temperature)
        )
        return response.choices[0].message.content.strip()

class GoogleLLMEngine(LLMEngine):
    def __init__(self, model_name: str, defaults=None):
        super().__init__(model_name, defaults)
        self.google_client = genai.Client(
          api_key=os.getenv("GEMINI_API_KEY"),
        )

    def generate_response(self, params: dict) -> str:
        response = self.google_client.models.generate_content(
          model=self.model_name,
          contents="",
          config=types.GenerateContentConfig(
              temperature=params.get("temperature", self.temperature),
              max_tokens=params.get("max_tokens", self.max_tokens),
              system_instruction= params.get("system_query", ""),
          )
        )
        return response.candidates[0].content.strip()


def get_engine(platform: str, model_name: str, defaults: dict = None):
    """
    Factory function to get the appropriate LLM engine based on the model name.
    """
    if platform == "groq":
        return GroqLLMEngine(model_name)
    elif platform == "google":
        return GoogleLLMEngine(model_name)
    else:
        raise ValueError(f"Unknown model name: {model_name}")