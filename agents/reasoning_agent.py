import json
from langchain.tools import tool
from langchain_community.llms import Ollama

llm = Ollama(model="llama3.2")

@tool
def summarize_comments(sentiment_path: str, bot_path: str, output_path: str):
    """Elemzi és összegzi a sentiment és bot detekció eredményeket."""
    with open(sentiment_path, "r", encoding="utf-8") as f:
        sentiment_data = json.load(f)

    with open(bot_path, "r", encoding="utf-8") as f:
        bot_data = json.load(f)

    prompt = f"""
    Az alábbi YouTube kommentek sentiment elemzésének és bot detekciójának összegzése:
    
    Sentiment eredmények: {json.dumps(sentiment_data, indent=2)}
    Bot detekciós eredmények: {json.dumps(bot_data, indent=2)}
    
    Kérlek, adj egy rövid összefoglalót a trendekről, valamint statisztikai elemzést!
    """

    summary = llm.invoke(prompt)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({"summary": summary}, f, indent=4)

    return f"Összegzés befejezve. Eredmény mentve: {output_path}"