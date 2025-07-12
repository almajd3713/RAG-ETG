import requests
from tqdm import tqdm
from langdetect import detect

API = "https://enterthegungeon.fandom.com/api.php"
HEADERS = {"User-Agent": "RAG-ing-Gungeoneer/1.0"}

def get_all_pages():
    pages = []
    cmcontinue = ""
    while True:
        params = {
            "action": "query",
            "list": "allpages",
            "aplimit": "max",
            "format": "json",
            "apcontinue": cmcontinue
        }
        response = requests.get(API, params=params, headers=HEADERS).json()
        pages.extend([p['title'] for p in response['query']['allpages']])
        if 'continue' not in response:
            break
        cmcontinue = response['continue']['apcontinue']
    return pages

def fetch_page_content(title):
    params = {
        "action": "query",
        "prop": "revisions",
        "rvprop": "content",
        "rvslots": "main",
        "formatversion": "2",
        "format": "json",
        "titles": title
    }
    response = requests.get(API, params=params, headers=HEADERS).json()
    pages = response.get("query", {}).get("pages", [])
    if pages and "revisions" in pages[0]:
        return pages[0]["revisions"][0]["slots"]["main"]["content"]
    return ""

pages = get_all_pages()
print(f"Found {len(pages)} pages.")

# Create directory if it doesn't exist, adjacent to the script
import os
base_dir = os.path.dirname(os.path.abspath(__file__))
os.makedirs(os.path.join(base_dir, "gungeon_pages"), exist_ok=True)
 
for title in tqdm(pages):
    content = fetch_page_content(title)
    if not content or len(content.strip()) < 20:
        continue  # Skip empty or too-short content
    if detect(content) != "en":
        continue
    path = os.path.join(base_dir, "gungeon_pages", f"{title.replace('/', '_')}.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
