import json
import os
import shutil
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

# Fájl elérési utak
LLM_MODEL = "llama3.2"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# --- Initialize Components (loaded once) ---
# Ensure Ollama is running and the model is pulled.
# You can set the host via the OLLAMA_HOST environment variable. It defaults to the value below if not set.

OLLAMA_BASE_URL = 'http://127.0.0.1:11500/'
OLLAMA_BASE_URL = os.getenv('OLLAMA_HOST', 'http://127.0.0.1:11500/')
try:
    llm = OllamaLLM(model=LLM_MODEL, base_url=OLLAMA_BASE_URL)
    embedding_function = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
except Exception as e:
    print(f"Failed to initialize LLM or Embeddings. Make sure Ollama is running and dependencies are installed. Error: {e}")
    llm = None
    embedding_function = None

# --- RAG Summarization ---

def summarize_with_rag(analysis_results):
    """
    Populates a vector store with analysis results and uses a RAG pipeline to generate a summary.
    """
    if not llm or not embedding_function:
        return "LLM or embedding model not initialized. Cannot generate summary."

    if not analysis_results:
        return "No comments were provided to analyze."

    documents = []
    for result in analysis_results:
        page_content = result.get("Comment", "")
        # Ensure metadata values are in a compatible format
        metadata = {
            "sentiment": str(result.get("Sentiment", "Unknown")),
            "polarity": float(result.get("Polarity", 0.0)),
            "prediction": str(result.get("Prediction", "Unknown"))
        }
        documents.append(Document(page_content=page_content, metadata=metadata))

    if not documents:
        return "No valid comments found to generate a summary."

    try:
        # 2. Create an in-memory vector store for this specific request.
        # This is faster and avoids data leakage between analyses without needing to delete files.
        vector_store = Chroma.from_documents(
            documents=documents,
            embedding=embedding_function
        )
        retriever = vector_store.as_retriever()
    except Exception as e:
        print(f"Error creating vector store: {e}")
        return "Failed to create the vector database for analysis."

    # 3. Define prompt and chain
    template = """
You are a helpful assistant who analyzes YouTube comments.
Based on the provided comments and their analysis (available as metadata), generate a concise, well-structured summary.
Your summary should address the following points:
1.  What are the main topics or themes people are discussing in the comments?
2.  What is the overall sentiment? Is it mostly positive, negative, or mixed?
3.  Are there any interesting patterns, recurring questions, or spam-like behavior (based on the 'bot' prediction metadata)?

Use only the information from the context below. Do not make things up. Also never ask for more information, and never say that you cant answer, or the informations or the context is too small.

Context:
{context}

Question: {question}

Answer:
"""
    prompt = ChatPromptTemplate.from_template(template)
    
    rag_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": prompt}
    )
    
    # 4. Invoke the chain to get the summary
    question = "Provide a summary of the YouTube comments based on the retrieved context."
    
    try:
        print("Querying RAG chain for summary...")
        result = rag_chain.invoke(question)
        print("Summary generated successfully.")
        return result.get("result", "Could not generate summary from LLM.")
    except Exception as e:
        print(f"Error during RAG summarization: {e}")
        return f"An error occurred while generating the summary with the LLM: {e}"

# --- Quantitative Summary (Kept for charts) ---

def load_json(file_path):
    if not os.path.exists(file_path):
        return []
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, FileNotFoundError):
        return []

def get_quantitative_summary(sentiment_path, bot_path):
    """
    Calculates a quantitative summary (counts) from the analysis result files.
    This is kept for the UI charts.
    """
    sentiment_results = load_json(sentiment_path)
    bot_results = load_json(bot_path)

    positive_count = sum(1 for r in sentiment_results if r.get('Sentiment') == 'Positive')
    negative_count = sum(1 for r in sentiment_results if r.get('Sentiment') == 'Negative')
    neutral_count = sum(1 for r in sentiment_results if r.get('Sentiment') == 'Neutral')

    bot_count = sum(1 for r in bot_results if r.get('Prediction') == 'bot')
    
    total_comments = len(sentiment_results)
    # Ensure human_count is not negative
    human_count = max(0, total_comments - bot_count)

    summary = {
        "total_comments": total_comments,
        "positive_comments": positive_count,
        "negative_comments": negative_count,
        "neutral_comments": neutral_count,
        "human_comments": human_count,
        "bot_comments": bot_count
    }
    return summary
