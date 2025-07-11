import json
import chromadb

with open("config.json", "r") as f:
    config = json.load(f)

collection_name = f"rag_etg_{config['embedding_model']['collection_name']}"
client = chromadb.PersistentClient(
    path="chroma_db",
)
collection = client.get_collection(
    name=collection_name
)

from sentence_transformers import SentenceTransformer
model = SentenceTransformer(config['embedding_model']['name'])

import os
import dotenv
dotenv.load_dotenv()
from groq import Groq
# Initialize Groq client
client = Groq(api_key=os.getenv("GROQ_KEY"))

def extract_query_info(query):
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
    response = client.chat.completions.create(
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
  
def extract_query_info_without_reformatting(query):
  return {
    "query": query,
  }

def get_query_text(query_text):
  if config['prepend_chunks_and_queries']:
    return f"Represent this sentence for searching relevant passages: {query_text}"
  else: 
    return query_text

def lookup_query(query_info):
    """
    Looks up the reformulated query in the ChromaDB collection.
    """
    query_text = get_query_text(query_info['query'])
    print(f"Querying for: {query_text}")
    emb = model.encode(query_text, normalize_embeddings=True)
    if not config.get('skip_reformatting', False) and "metadata" in query_info:
      where_clause = {
        "$or": [
          {"section": query_info["metadata"]["section"]},
          {"title": query_info["metadata"]["item"]}
        ]
      }
    else:
      where_clause = None

    results = collection.query(
      query_embeddings=[emb],
      n_results=config['retrieval_settings']['top_k'],
      where=where_clause
    )
    
    threshold = config['retrieval_settings']['similarity_threshold']
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

def get_llm_answer(query, context_array):
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

if __name__ == "__main__":
    if config['is_cli']:
        user_query = input("Enter your query: ")
    else:
        user_query = "what items synergize with the metronome? List them all and their effects."

    if not config['skip_reformatting']:
      query_info = extract_query_info(user_query)
    else:
      query_info = extract_query_info_without_reformatting(user_query)
    print("Extracted Query Information:")
    print(query_info)

    results = lookup_query(query_info)
    print("Top K Result scores:")
    for i, doc in enumerate(results):
        print(f"{doc['metadata']['id']} score: {doc['distance']}")
    context_array = [doc['document'] for doc in results]

    llm_answer = get_llm_answer(user_query, context_array)
    print("LLM Answer:")
    print(llm_answer)