import json
import ollama
import os

# Fájl elérési utak
DATA_DIR = "data"

# JSON betöltése
def load_json(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Hiba: {file_path} nem található!")
    
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)

# Prompt generálás a Llama 3.2 számára
def generate_prompt(comments, sentiment_results, bot_results):
    prompt = f"""
    Analyze the following YouTube comments and their metadata: 
    - What are the main topics that emerge? 
    - Is the overall sentiment positive, negative, or mixed? 
    - How credible are these comments (based on bot detection)? 
    - Are there any recurring trends or patterns?
    
    Comments:
    {comments}

    Sentiment analysis results:
    {sentiment_results}

    Bot detection results:
    {bot_results}

    Please provide a well-structured summary!
    """
    return prompt

# Ollama LLM hívása
def query_llama3(prompt):
    response = ollama.chat(model="llama3.2", messages=[{"role": "user", "content": prompt}])
    return response["message"]["content"]

# Eredmény mentése
def save_summary(summary, output_path):
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({"summary": summary}, f, indent=4)
    print(f"Összegzés mentve: {output_path}")

# Reasoning függvény agent számára
def summarize_comments(sentiment_path, bot_path, output_path="data/summary.json"):
    comments = load_json(os.path.join(DATA_DIR, "comments.json"))
    sentiment_results = load_json(sentiment_path)
    bot_results = load_json(bot_path)

    prompt = generate_prompt(comments, sentiment_results, bot_results)
    summary = query_llama3(prompt)
    
    save_summary(summary, output_path)
    return summary  # Visszatér a summary szöveggel, hogy az agent tudja használni

#teszthez
if __name__ == "__main__":
    summarize_comments("data/sentiment_results.json", "data/bot_detection_results.json")
    print("Reasoning folyamat sikeresen lefutott!")

