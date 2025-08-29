import os
import json
#from models.sentiment_analysis import sentiment_analysis as analyze_sentiment
from models.sentiment_analysis_bertmini import sentiment_analysis as analyze_sentiment
from models.reasoning import summarize_comments
import json
import os

# Adatmappa
DATA_DIR = "data"

# Fájl elérési utak FRISSÍTVE
COMMENTS_PATH = os.path.join(DATA_DIR, "comments.json")
SENTIMENT_PATH = os.path.join(DATA_DIR, "sentiment_results.json")
SUMMARY_PATH = os.path.join(DATA_DIR, "summary.json")

# JSON fájlok beolvasására szolgáló függvény
def load_json(filepath):
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Hiba a(z) {filepath} beolvasásakor: {e}")
        return None  # Ha nincs fájl, None értéket adunk vissza

def run_analysis():
    print("Social Media Filter elindult...\n")

    # Kommentek beolvasása JSON formátumban
    print("Loading comments...")
    comments_data = load_json(COMMENTS_PATH)
    if comments_data is None:
        print("Nem sikerült betölteni a kommenteket!")
        return
    print("Comments loaded.")

    # Sentiment analízis futtatása
    print("Running sentiment analysis...")
    try:
        analyze_sentiment(COMMENTS_PATH, SENTIMENT_PATH)
        sentiment_result = load_json(SENTIMENT_PATH)
        print("\nSentiment Analízis eredménye:", sentiment_result)
    except Exception as e:
        print("\nHiba történt a Sentiment Analízis során:", str(e))
    print("Sentiment analysis finished.")

    # Eredmények beolvasása összegzéshez
    sentiment_data = load_json(SENTIMENT_PATH)

    if sentiment_data is None:
        print("Nem sikerült betölteni az előző lépések eredményeit!")
        return

    # Reasoning (összegzés) futtatása
    print("Running summarization...")
    summary_result = summarize_comments(SENTIMENT_PATH) # bot results path is not used
    print("\nÖsszegzés eredménye:", summary_result)
    print("Summarization finished.")

    # Összegzés mentése
    with open(SUMMARY_PATH, 'w', encoding='utf-8') as f:
        json.dump(summary_result, f, ensure_ascii=False, indent=4)

    print("\nMinden művelet sikeresen lefutott!")

def main():
    run_analysis()

if __name__ == "__main__":
    main()
