import os
from langchain.agents import initialize_agent
from langchain.tools import Tool
from langchain.agents import AgentType
from agents.sentiment_agent import analyze_sentiment
from agents.bot_agent import detect_bots
from agents.reasoning_agent import summarize_comments
import ollama

# Adatmappa
DATA_DIR = "data"

# Fájl elérési utak
COMMENTS_PATH = os.path.join(DATA_DIR, "comments.json")
SENTIMENT_PATH = os.path.join(DATA_DIR, "sentiment.json")
BOT_RESULTS_PATH = os.path.join(DATA_DIR, "bot_results.json")
SUMMARY_PATH = os.path.join(DATA_DIR, "summary.json")

#ollama start
llm = ChatOllama(model="llama3")

# LangChain agentek listája
tools = [
    Tool(
        name="SentimentAnalyzer",
        func=analyze_sentiment,
        description="Elemzi a kommentek sentimentjét (pozitív, negatív, semleges)."
    ),
    Tool(
        name="BotDetector",
        func=detect_bots,
        description="Meghatározza, hogy egy kommentet bot vagy ember írta-e."
    ),
    Tool(
        name="Summarizer",
        func=summarize_comments,
        description="Összegzi a sentiment és bot detekció eredményeit, statisztikát adva."
    ),
]

# AgentExecutor létrehozása
agent_executor = initialize_agent(
    tools=tools,
    llm=llm,  # Itt adjuk meg az LLM-et
    agent="zero-shot-react-description",
    verbose=True
)

def main():
    print("🚀 Social Media Filter elindult LangChain-nel...")

    # 1️⃣ Sentiment analízis
    agent_executor.invoke({"input": f"{COMMENTS_PATH}, {SENTIMENT_PATH}"})

    # 2️⃣ Bot detekció
    agent_executor.invoke({"input": f"{COMMENTS_PATH}, {BOT_RESULTS_PATH}"})

    # 3️⃣ Reasoning (összegzés)
    agent_executor.invoke({"input": f"{SENTIMENT_PATH}, {BOT_RESULTS_PATH}, {SUMMARY_PATH}"})

    print("\n🎉 Minden LangChain agent sikeresen lefutott!")

if __name__ == "__main__":
    main()
