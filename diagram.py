import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os

def generate_diagram(data_path, output_path):
    # JSON beolvasása
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # DataFrame-re alakítás
    df = pd.DataFrame(data)

    # Sentiment kategóriák összesítése
    sentiment_counts = df['Sentiment'].value_counts().reset_index()
    sentiment_counts.columns = ['Sentiment', 'Count']

    # Kördiagram rajzolása Seaborn színekkel
    plt.figure(figsize=(8, 8))
    colors = sns.color_palette(['green', 'grey', 'red'])
    plt.pie(
        sentiment_counts['Count'],
        labels=sentiment_counts['Sentiment'],
        colors=colors,
        autopct='%1.1f%%',
        startangle=140
    )
    plt.title('Sentiment Distribution Pie Chart')
    
    # Diagram mentése
    plt.savefig(output_path)
    plt.close()

if __name__ == '__main__':
    data_path = os.path.join('data', 'sentiment_results.json')
    output_path = os.path.join('static', 'sentiment_diagram.png')
    if not os.path.exists('static'):
        os.makedirs('static')
    generate_diagram(data_path, output_path)
