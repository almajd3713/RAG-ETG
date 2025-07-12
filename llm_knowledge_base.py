import json

import chromadb
from chat_history import ChatHistory
from llm_embedder import LLMEmbedder

# --- SETUP ---
client = chromadb.PersistentClient(
    path="chroma_db",
)
# ----------------

class KnowledgeBase:
	def __init__(self, embedder: LLMEmbedder, chat_history: ChatHistory, config: dict,  logger=None):
		self.config = config
		self.embedder = embedder
		self.chat_history = chat_history
		self.collection = client.get_collection(
			name=f"rag_etg_{config['embedding_model']['collection_name']}"
		)
		self.logger = logger

	def embed(self, text):
		return self.embedder.embed(text)

	def query(self, query, conversation_focus=None):
		"""
		Process the user query to extract relevant information and retrieve context.
		"""
		query_info = self.embedder.extract_query_info(query, self.chat_history.get_chat(), conversation_focus)
		if self.logger: self.logger.info(f"Query Info: {json.dumps(query_info, indent=2)}")
  
		# Check if the query has enough context to skip lookup, avoids bloating context with unnecessary information.
		if self.check_if_in_context(query_info['query']):
			if self.logger: self.logger.info("Query has enough context, skipping lookup.")
			return query_info['query'], None
  
		# Query the ChromaDB collection
		context_raw = self._query(query_info)
		context_array = [doc['document'] for doc in context_raw]
		if self.logger: self.logger.info(f"Context Array: {context_array}")
		if not context_array:
			return "Not enough information in the context to answer this question."

		return query_info['query'], context_array

	def _query(self, query_info):
			"""
			Looks up the reformulated query in the ChromaDB collection.
			"""
			query_text = self.embedder.get_query_text(query_info['query'])
			emb = self.embedder.embed(query_text)
			if not self.config.get('skip_reformatting', False) and "metadata" in query_info:
				where_clause = {
					"$or": [
						{"section": query_info["metadata"]["section"]},
						{"title": query_info["metadata"]["item"]}
					]
				}
			else:
				where_clause = None

			results = self.collection.query(
				query_embeddings=[emb],
				n_results=self.config['retrieval_settings']['top_k'],
				where=where_clause
			)

			if self.logger: 
				for id, distance in zip(results['ids'][0], results['distances'][0]):
					self.logger.info(f"ID: {id}, Distance: {distance}")

			threshold = self.config['retrieval_settings']['similarity_threshold']
			if threshold is not None:
				# Extract raw fields
				documents = results['documents'][0]
				distances = results['distances'][0]
				metadatas = results['metadatas'][0]

				# Filter based on distance
				filtered_docs = [
						{"document": doc, "metadata": meta, "distance": dist} for doc, dist, meta in zip(documents, distances, metadatas) if dist <= threshold
				]
				return filtered_docs
			return results
	def check_if_in_context(self, reformulated_query):
		"""
		Check if the query has enough context to skip lookup, avoids bloating context with unnecessary information.
		"""
		if self.logger: self.logger.info(f"Checking if context is enough for query: {reformulated_query}, current context length: {len(self.chat_history.context_history)}")
		if not self.chat_history: return False
		if not len(self.chat_history.context_history):
			if self.logger: self.logger.info("No context available, lookup is necessary.")
			return False
  
		scores = [
			self._cosine_distance(self.embed(reformulated_query), self.embed(item)) for item in self.chat_history.context_history
		]
		if self.logger: self.logger.info(f"Context Scores: {scores}")
		has_enough_context = any(score < self.config['chat_history']['lookup_score_threshold'] for score in scores)
		if self.logger: self.logger.info(f"Has enough context: {has_enough_context}")
		return has_enough_context

	def _cosine_distance(self, vec1, vec2):
		"""
		Compute the cosine similarity between two vectors.
		"""
		if len(vec1) != len(vec2):
			raise ValueError("Vectors must be of the same length")

		dot_product = sum(a * b for a, b in zip(vec1, vec2))
		magnitude1 = sum(a ** 2 for a in vec1) ** 0.5
		magnitude2 = sum(b ** 2 for b in vec2) ** 0.5

		if magnitude1 == 0 or magnitude2 == 0:
			return 0.0
		output = dot_product / (magnitude1 * magnitude2)
		return 1 - output
