import os
import mwparserfromhell
import logging
import datetime

# Configure logging
log_dir = "logs/prepare_data"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
log_file = os.path.join(log_dir, f"{current_time}.log")
logging.basicConfig(filename=log_file, level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

def clean_redirect_files(directory="gungeon_pages"):
  """
  Removes files from the specified directory if their only content
  is a line starting with #REDIRECT.
  """
  logging.info(f"Starting to clean redirect files in directory: {directory}")
  if not os.path.isdir(directory):
    logging.error(f"Directory '{directory}' not found.")
    return

  for filename in os.listdir(directory):
    filepath = os.path.join(directory, filename)
    if os.path.isfile(filepath):
      try:
        with open(filepath, 'r', encoding='utf-8') as f:
          lines = f.readlines()
        
        if len(lines) == 1 and lines[0].strip().upper().startswith("#REDIRECT"):
          os.remove(filepath)
          logging.info(f"Removed redirect file: {filepath}")
      except Exception as e:
        logging.error(f"Error processing file {filepath}: {e}")
  logging.info(f"Finished cleaning redirect files in directory: {directory}")

import os
import mwparserfromhell
import json
import re
from langdetect import detect, LangDetectException

def clean_links_and_templates(text):
    # Preserve links in format: display_text (link_target)
    text = re.sub(r"\[\[([^\|\]]+)\|([^\]]+)\]\]", r"\2 (\1)", text)  # [[target|text]] -> text (target)
    text = re.sub(r"\[\[([^\]]+)\]\]", r"\1", text)  # [[target]] -> target
    text = re.sub(r"\{\{[^\}]+\}\}", "", text)  # remove templates
    text = re.sub(r"\[\[Category:[^\]]+\]\]", "", text)
    return text.strip()


def parse_wikitext_files(input_directory="gungeon_pages", output_directory="parsed_gungeon_pages_json"):
    logging.info(f"Starting to parse wikitext files from '{input_directory}' to '{output_directory}'.")
    if not os.path.isdir(input_directory):
        logging.error(f"Input directory '{input_directory}' not found.")
        return

    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
        logging.info(f"Created output directory: {output_directory}")

    for filename in os.listdir(input_directory):
        input_filepath = os.path.join(input_directory, filename)
        if os.path.isfile(input_filepath) and filename.endswith(".txt"):
            try:
                with open(input_filepath, 'r', encoding='utf-8') as f:
                    content = f.read()

                wikicode = mwparserfromhell.parse(content)

                # Extract title from filename
                title = filename.replace(".txt", "").replace("_", " ")

                # Extract infobox
                infobox = None
                for template in wikicode.filter_templates():
                    if "infobox" in template.name.lower():
                        infobox = {param.name.strip(): clean_links_and_templates(str(param.value)) for param in template.params}
                        break

                # Extract and group sections
                sections = []
                current_section = {"heading": "Summary", "content": []}

                for node in wikicode.nodes:
                  if isinstance(node, mwparserfromhell.nodes.Heading):
                      if current_section["content"]:
                          sections.append(current_section)
                      current_section = {"heading": str(node.title).strip(), "content": []}

                  elif isinstance(node, mwparserfromhell.nodes.Text):
                      lines = str(node).split("\n")
                      for line in lines:
                          if line.strip():
                              current_section["content"].append(line.strip())

                  elif isinstance(node, mwparserfromhell.nodes.Wikilink):
                      target = str(node.title).strip()
                      text = str(node.text).strip() if node.text else target
                      current_section["content"].append(f"{text} ({target})")

                  elif isinstance(node, mwparserfromhell.nodes.Template):
                      continue  # only keep templates for infobox up top

                # Add last section
                if current_section["content"]:
                    sections.append(current_section)

                # Save as JSON
                result = {
                    "title": title,
                    "infobox": infobox if infobox else {},
                    "sections": sections
                }

                output_filepath = os.path.join(output_directory, filename.replace(".txt", ".json"))
                with open(output_filepath, 'w', encoding='utf-8') as out:
                    json.dump(result, out, indent=2, ensure_ascii=False)

                logging.info(f"Processed '{input_filepath}' -> '{output_filepath}'")

            except Exception as e:
                logging.error(f"Error processing file {input_filepath}: {e}")
    logging.info(f"Finished parsing wikitext files.")

def flatten_infobox_text(infobox_dict):
    lines = []
    for key, value in infobox_dict.items():
        val = str(value).strip()
        lines.append(f"{key}: {val}")
    return "\n".join(lines)

def flatten_section(title, section, infobox=None):
    section_title = section["heading"]
    lines = section.get("content", [])
    clean_text = " ".join(line.strip() for line in lines if line.strip())

    # Add infobox text at the top if section is Summary
    if section_title.lower() == "summary" and infobox:
        infobox_text = flatten_infobox_text(infobox)
        full_text = f"{section_title}\n{infobox_text}\n\n{clean_text}"
    else:
        full_text = f"{section_title}\n{clean_text}"

    return {
        "id": f"{title}:{section_title}",
        "text": full_text,
        "meta": {
            "title": title,
            "section": section_title
        }
    }

def flatten_page(page):
    title = page["title"]
    infobox = page.get("infobox", {})
    sections = page.get("sections", [])
    chunks = []

    for section in sections:
        chunk = flatten_section(title, section, infobox=infobox)
        chunks.append(chunk)

    return chunks

def load_and_flatten_pages(input_dir):
    logging.info(f"Starting to load and flatten pages from directory: {input_dir}")
    chunks = []
    for filename in os.listdir(input_dir):
        if filename.endswith(".json"):
            path = os.path.join(input_dir, filename)
            try:
                with open(path, "r", encoding="utf-8") as f:
                    page = json.load(f)
                chunks.extend(flatten_page(page))
                logging.debug(f"Successfully flattened {filename}")
            except Exception as e:
                logging.error(f"Error flattening file {path}: {e}")
    logging.info(f"Finished loading and flattening pages. Total chunks: {len(chunks)}")
    return chunks

def is_english(chunk):
    meta_title = chunk.get("meta", {}).get("title", "")
    text = chunk.get("text", "")
    if re.search(r'\bzh\b', text) or re.search(r'\bzh\b', meta_title):
        return False
    try:
        lang = detect(text)
        return lang == "en"
    except LangDetectException:
        return False

def filter_english_chunks(chunks):
    english_chunks = []
    for chunk in chunks:
        if is_english(chunk):
            english_chunks.append(chunk)
        else:
            logging.info(f"Dropped non-English chunk: {chunk.get('id', '')}")
    return english_chunks

if __name__ == "__main__":
  logging.info("Starting cleanup...")
  clean_redirect_files()
  logging.info("Cleanup finished.")
  logging.info("Starting parsing...")
  parse_wikitext_files()
  logging.info("Parsing finished.")
  logging.info("Starting flattening...")
  chunks = load_and_flatten_pages("parsed_gungeon_pages_json")
  logging.info("Flattening finished.")
  logging.info(f"Total chunks: {len(chunks)}")
  logging.info("Filtering English chunks...")
  chunks = filter_english_chunks(chunks)
  logging.info(f"Total English chunks: {len(chunks)}")
  logging.info("Saving chunks to JSON file...")
  with open("all_chunks.json", "w", encoding="utf-8") as f:
    json.dump(chunks, f, indent=2, ensure_ascii=False)
  logging.info("Chunks saved to all_chunks.json.")
  logging.info("Script completed successfully.")

