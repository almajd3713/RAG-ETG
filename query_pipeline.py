import json

with open("config.json", "r") as f:
    config = json.load(f)

from llm_manager import LLMManager

if __name__ == "__main__":
    llm_manager = LLMManager()

    if config['is_cli']:
        user_query = input("Enter your query: ")
    else:
        user_query = "What are the names of the synergies that the metronome has?"
        
    llm_answer = llm_manager.query(user_query)
    print("LLM Answer:")
    print(llm_answer)