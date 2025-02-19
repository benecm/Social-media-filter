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

def get_youtube_comments(video_url, api_key='AIzaSyCYzkS4z6JBJ8fkvsSiLIJdTGj83URNNRc'):
    video_id = extract_video_id(video_url)
    if not video_id:
        print("Érvénytelen YouTube link!")
        return []
    
    youtube = build("youtube", "v3", developerKey=api_key) #api kapcsolat letrehozasa
    comments = []
    
    request = youtube.commentThreads().list(    #elsodleges kommenteket kerjuk le
        part="snippet",         #kommentekhez tartozo reszletek
        videoId=video_id,       #video kivalasztasa
        textFormat="plainText", #szovegkent kapjuk vissza nem htmlben
        maxResults=10000        #lekert kommentek szama
    )
    #a request egy JSON file, az alabbi fugveny bejarja es kiirja teszt jelleggel
    while request:
        response = request.execute()
        for item in response.get("items", []):
            comment = item["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
            comments.append(comment)
        
        request = youtube.commentThreads().list_next(request, response)
    
    return comments

#kapott kommenteket kimenti json-be, ha mar letzik hozzafuzi
def save_comments_to_json(comments, filename="comments.json"):
    try:
        with open(filename, "r", encoding="utf-8") as f:
            existing_comments = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        existing_comments = []
    
    existing_comments.extend(comments)
    
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(existing_comments, f, ensure_ascii=False, indent=4)
    print(f"Kommentek hozzáfűzve: {filename}")

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