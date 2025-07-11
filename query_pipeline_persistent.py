import json

with open("config.json", "r") as f:
    config = json.load(f)

from llm_manager import LLMManager

if __name__ == "__main__":
  llm_manager = LLMManager(config, True, log_dir=config.get('log_dir', None))
  
  try:
    while True:
      user_input = input("Enter your query (type 'exit' to quit): ")
      if user_input.strip().lower() == "exit":
        print("Exiting.")
        break
      response = llm_manager.query(user_input)
      print("LLM Response:", response)
  except KeyboardInterrupt:
    print("\nKeyboard interrupt received. Exiting.")
  