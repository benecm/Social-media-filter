import os
import json
from langchain.agents import initialize_agent, AgentType
from langchain.tools import Tool
from agents.sentiment_agent import analyze_sentiment
from agents.bot_agent import detect_bots
from agents.reasoning_agent import summarize_comments
from langchain_ollama import ChatOllama

# Adatmappa
DATA_DIR = "data"

# Fájl elérési utak FRISSÍTVE
COMMENTS_PATH = os.path.join(DATA_DIR, "comments.json")
SENTIMENT_PATH = os.path.join(DATA_DIR, "sentiment_results.json")
BOT_RESULTS_PATH = os.path.join(DATA_DIR, "bot_detection_results.json")
SUMMARY_PATH = os.path.join(DATA_DIR, "summary.json")

# Ollama LLM inicializálása - NYELVI KORLÁTOZÁSSAL
llm = ChatOllama(model="llama3.2", system="Respond in English only.")

# JSON fájlok beolvasására szolgáló függvény
def load_json(filepath):
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Hiba a(z) {filepath} beolvasásakor: {e}")
        return None  # Ha nincs fájl, None értéket adunk vissza

# LangChain agentek listája
tools = [
    Tool(
        name="SentimentAnalyzer",
        func=lambda input_data: analyze_sentiment(input_data, SENTIMENT_PATH),
        description="Analyzes comment sentiment (positive, negative, neutral)."
    ),
    Tool(
        name="BotDetector",
        func=lambda input_data: detect_bots(input_data, BOT_RESULTS_PATH),
        description="Determines whether a comment is written by a bot or a human."
    ),
    Tool(
        name="Summarizer",
        func=lambda input_data: summarize_comments(input_data, SUMMARY_PATH),
        description="Summarizes sentiment and bot detection results, providing statistics."
    ),
]

# AgentExecutor létrehozása - OUTPUT PARSING ERROR KEZELÉSSEL
agent_executor = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    handle_parsing_errors=True  # Az LLM válaszainak automatikus újrapróbálása hibás formátum esetén
)

def main():
    print("Social Media Filter elindult LangChain-nel...\n")

    # Kommentek beolvasása JSON formátumban
    comments_data = load_json(COMMENTS_PATH)
    if comments_data is None:
        print("Nem sikerült betölteni a kommenteket!")
        return

    # Sentiment analízis futtatása
    try:
        sentiment_result = agent_executor.invoke({"input": comments_data})
        print("\nSentiment Analízis eredménye:", sentiment_result)
    except Exception as e:
        print("\nHiba történt a Sentiment Analízis során:", str(e))

    # Bot detekció futtatása
    try:
        bot_result = agent_executor.invoke({"input": comments_data})
        print("\nBot Detekció eredménye:", bot_result)
    except Exception as e:
        print("\nHiba történt a Bot Detekció során:", str(e))

    # Eredmények beolvasása összegzéshez
    sentiment_data = load_json(SENTIMENT_PATH)
    bot_data = load_json(BOT_RESULTS_PATH)

    if sentiment_data is None or bot_data is None:
        print("Nem sikerült betölteni az előző lépések eredményeit!")
        return

    # Reasoning (összegzés) futtatása
    try:
        summary_result = agent_executor.invoke({"input": {"sentiment": sentiment_data, "bots": bot_data}})
        print("\nÖsszegzés eredménye:", summary_result)
    except Exception as e:
        print("\nHiba történt az Összegzés során:", str(e))

    print("\nMinden LangChain agent sikeresen lefutott!")

if __name__ == "__main__":
    main()
