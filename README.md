# YouTube Comment Analyzer

This project is a web application that allows you to analyze the sentiment of comments on any YouTube video. It scrapes the comments, performs sentiment analysis, and provides a summary of the results.

## Features

- **YouTube Comment Scraping:** Fetches a specified number of comments from any YouTube video.
- **Sentiment Analysis:** Classifies comments as positive, negative, or neutral.
- **Results Summarization:** Provides a summary of the sentiment analysis results, including the total number of comments and the count of positive, negative, and neutral comments.
- **Data Visualization:** Generates a pie chart to visualize the distribution of sentiments.

## How It Works

The application is built with a Flask backend and a simple HTML frontend. Here's a breakdown of the workflow:

1.  **Input:** The user provides a YouTube video URL and the number of comments to analyze through a simple web interface.
2.  **Scraping:** The application uses the YouTube Data API to fetch the comments from the specified video.
3.  **Analysis:** The comments are then processed by a sentiment analysis model built with `TextBlob`.
4.  **Summarization:** The application generates a summary of the analysis, counting the number of positive, negative, and neutral comments.
5.  **Output:** The sentiment analysis results and the summary are displayed to the user on the web page. A pie chart visualizing the sentiment distribution is also generated.

## How to Use

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/your-repo-name.git
    cd your-repo-name
    ```

2.  **Install the dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the application:**
    ```bash
    python webapp.py
    ```

4.  **Open your browser** and navigate to `http://127.0.0.1:5000/`.

5.  **Enter a YouTube video URL** and the number of comments you want to analyze, then click "Analyze".

## Technologies Used

- **Backend:** Python, Flask
- **Frontend:** HTML, JavaScript
- **Libraries:**
    - `google-api-python-client` (for YouTube data)
    - `textblob` (for sentiment analysis)
    - `pandas` (for data manipulation)
    - `matplotlib` (for plotting)
    - `seaborn` (for enhanced visualizations)
