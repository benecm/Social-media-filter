import os
from langchain.agents import initialize_agent
from langchain.tools import Tool
from langchain.agents import AgentType
from agents.sentiment_agent import sentiment_tool
from agents.bot_agent import bot_detection_tool
from agents.reasoning_agent import reasoning_tool

# Adatmappa
DATA_DIR = "data"

# Fájl elérési utak
COMMENTS_PATH = os.path.join(DATA_DIR, "comments.json")
SENTIMENT_PATH = os.path.join(DATA_DIR, "sentiment.json")
BOT_RESULTS_PATH = os.path.join(DATA_DIR, "bot_results.json")
SUMMARY_PATH = os.path.join(DATA_DIR, "summary.json")

# LangChain agentek listája
tools = [
    Tool(name="SentimentAnalyzer", func=sentiment_tool),
    Tool(name="BotDetector", func=bot_detection_tool),
    Tool(name="Summarizer", func=reasoning_tool),
]

# AgentExecutor létrehozása
agent_executor = initialize_agent(
    tools=tools,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,  # A LangChain egyik működési módja
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
