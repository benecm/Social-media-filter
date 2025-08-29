from textblob import TextBlob
import json
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import json
import re #regex megoldas
from googleapiclient.discovery import build

def extract_video_id(video_url):
    match = re.search(r"(?:v=|\/)([0-9A-Za-z_-]{11})", video_url)
    return match.group(1) if match else None

def remove_emojis(data):
    emoj = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
        u"\U00002500-\U00002BEF"  # chinese char
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        u"\U0001f926-\U0001f937"
        u"\U00010000-\U0010ffff"
        u"\u2640-\u2642" 
        u"\u2600-\u2B55"
        u"\u200d"
        u"\u23cf"
        u"\u23e9"
        u"\u231a"
        u"\ufe0f"  # dingbats
        u"\u3030"
                      "]+", re.UNICODE)
    return re.sub(emoj, '', data)

def get_youtube_comments(video_url, api_key='AIzaSyCYzkS4z6JBJ8fkvsSiLIJdTGj83URNNRc', max_results=100):
    video_id = extract_video_id(video_url)
    if not video_id:
        print("Érvénytelen YouTube link!")
        return []

    youtube = build("youtube", "v3", developerKey=api_key)
    comments = []
    
    request = youtube.commentThreads().list(
        part="snippet",
        videoId=video_id,
        textFormat="plainText",
        maxResults=min(int(max_results), 10000)  # Biztonsági ellenőrzés
    )

    while request and len(comments) < int(max_results):
        response = request.execute()
        for item in response.get("items", []):
            comment = item["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
            comment = remove_emojis(comment)
            if(comment != ""):
                comments.append(comment)
            if len(comments) >= int(max_results):
                break
        
        if len(comments) >= int(max_results):
            break

        request = youtube.commentThreads().list_next(request, response)
    
    return comments

#kapott kommenteket kimenti json-be, ha mar letzik hozzafuzi
def save_comments_to_json(comments, filename="comments.json"):
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(comments, f, ensure_ascii=False, indent=4)
    print(f"Kommentek mentve: {filename}")

def analyze_comments(filename="comments.json"):
    try:
        with open(filename, "r", encoding="utf-8") as f:
            comments = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        print("Nem található megfelelő JSON fájl!")
        return
    
    df = pd.DataFrame(comments, columns=["Comment"])
    df["Length"] = df["Comment"].apply(len)
    
    print("Példa kommentek:")
    print(df.head())
    
    print("\nStatisztikák:")
    print(df.describe())
    
    plt.figure(figsize=(10, 5))
    plt.hist(df["Length"], bins=30, edgecolor='black')
    plt.xlabel("Karakterek száma")
    plt.ylabel("Gyakoriság")
    plt.title("Komment hosszúságok eloszlása")
    #plt.show()

def sentiment_analysis(filename="comments.json", output_filename="sentiment_results.json"):
    try:
        with open(filename, "r", encoding="utf-8") as f:
            comments = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        print("Nem található megfelelő JSON fájl!")
        return
    
    df = pd.DataFrame(comments, columns=["Comment"])
    df["Polarity"] = df["Comment"].apply(lambda x: TextBlob(x).sentiment.polarity)
    df["Sentiment"] = df["Polarity"].apply(lambda x: "Positive" if x > 0 else "Negative" if x < 0 else "Neutral")
    
    sentiment_results = df.to_dict(orient="records")
    with open(output_filename, "w", encoding="utf-8") as f:
        json.dump(sentiment_results, f, ensure_ascii=False, indent=4)
    
    print("Sentiment elemzés eredménye mentve:", output_filename)
    
    plt.figure(figsize=(8, 5))
    df["Sentiment"].value_counts().plot(kind="bar", color=['green', 'red', 'gray'])
    plt.xlabel("Sentiment")
    plt.ylabel("Frekvencia")
    plt.title("Kommentek sentiment eloszlása")
    #plt.show()