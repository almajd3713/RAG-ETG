import json
from sentence_transformers import SentenceTransformer

# --- SETUP ---
with open("config.json", "r") as f:
    config = json.load(f)
# ----------------

class LLMEmbedder:
	def __init__(self, model_name, groq_client):
			self.model = SentenceTransformer(model_name)
			self.groq_client = groq_client

	def embed(self, text):
			return self.model.encode(
					text, normalize_embeddings=True,
					device='cuda', batch_size=64, show_progress_bar=False
			)

	def extract_query_info(self, query):
			"""
			Extracts relevant information from the user query.
			"""
			if config.get('skip_reformatting', False):
					return self._extract_query_info_without_reformatting(query)
			else:
					return self._extract_query_info_with_reformatting(query)
	def _extract_query_info_with_reformatting(self, query):
			"""
			Extracts relevant information from the user query.
			"""
			system_prompt = f"""
			You are a query rewriter for a retrieval system in a roguelike videogame context. Your task is to reformulate user queries by removing filler words and making them short, specific, and semantically equivalent. Do not add new information. Do not change the meaning.

			Always respond with this JSON format:
			{{
			"query": "<reformulated query>",
			"metadata": {{
					"section": "<reformulated section>",
					"item": "<reformulated item>",
					}}
			}}
			Usage of the words in this list is highly recommended if possible: "Summary", "Notes", "Trivia", "Effects", "Bugs", "Behavior", "Gallery", "Changes", "Synergies", "Strategy", "Behaviour", "Items", "Quotes", "Tips", "Story", "Past Kill", "Guns", "Exit the Gungeon", "Bug Fixes", "Video", "Hotfix 1", "Forge", "Major", "Enemies", "Hollow", "Black Powder Mine", "Gungeon Proper", "Keep of the Lead Lord", "Boss", "Enter the Gungeon", "Jetpack Variant", "Improvements/Balance Changes".

			Examples:

			User: "Can you tell me what the effects of the Gunzheng gun are?"
			Response:
			{{
			"query": "Gunzheng effects",
			"metadata": {{
					"section": "Effects",
					"item": "Gunzheng"
			}}
			}}

			User: "What is the lore and background of the Makarov weapon?"
			Response:
			{{
			"query": "Makarov trivia",
			"metadata": {{
					"section": "Trivia",
					"item": "Makarov"
					}}
			}}
			"""
			response = self.groq_client.chat.completions.create(
					model="llama-3.1-8b-instant",
					messages=[
							{"role": "system", "content": system_prompt},
							{"role": "user", "content": query}
					],
					max_tokens=100,
					temperature=0.0
			)

			if response.choices:
					return json.loads(response.choices[0].message.content.strip())
			return None
    
	def _extract_query_info_without_reformatting(self, query):
			return {
					"query": query,
			}
    
	def get_query_text(self, query_text):
		if config['prepend_chunks_and_queries']:
			return f"Represent this sentence for searching relevant passages: {query_text}"
		else: 
			return query_text