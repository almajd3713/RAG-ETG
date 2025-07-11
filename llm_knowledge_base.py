import json

import chromadb
from llm_embedder import LLMEmbedder
from chromadb import Collection

# --- SETUP ---
client = chromadb.PersistentClient(
    path="chroma_db",
)
# ----------------

class KnowledgeBase:
	def __init__(self, embedder: LLMEmbedder, config: dict):
		self.config = config
		self.embedder = embedder
		self.collection = client.get_collection(
			name=config['embedding_model']['collection_name']
		)

	def embed(self, text):
		return self.embedder.embed(text)

	def query(self, query):
		"""
		Process the user query to extract relevant information and retrieve context.
		"""
		query_info = self.embedder.extract_query_info(query)
		context_raw = self._query(query_info)
		context_array = [doc['document'] for doc in context_raw]

		if not context_array:
			return "Not enough information in the context to answer this question."

		return context_array

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