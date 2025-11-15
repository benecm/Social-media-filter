import json
import os
import pandas as pd
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# Import the analysis functions you want to test.
from models.sentiment_analysis_textblob import sentiment_analysis as analyze_sentiment
from models.bot_detection_modell import run_bot_detection

# Define constants for file paths
DATA_DIR = "data"
GROUND_TRUTH_PATH = os.path.join(DATA_DIR, "ground_truth.json")
PERFORMANCE_REPORT_PATH = os.path.join(DATA_DIR, "performance_report.json")

# Temporary files for the analysis functions
TEMP_COMMENTS_PATH = os.path.join(DATA_DIR, "temp_test_comments.json")
TEMP_SENTIMENT_PATH = os.path.join(DATA_DIR, "temp_sentiment_results.json")
TEMP_BOT_PATH = os.path.join(DATA_DIR, "temp_bot_results.json")

def load_ground_truth(filepath):
    """Loads the ground truth data from a JSON file."""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error loading ground truth file {filepath}: {e}")
        print("Please create a 'ground_truth.json' file in the 'data' directory.")
        print("It should be a list of objects, each with 'Comment', 'Sentiment', and 'Prediction' keys.")
        return None

def calculate_metrics(y_true, y_pred, labels):
    """Calculates and returns a dictionary of classification metrics."""
    # Note: RMSE (Root Mean Squared Error) is used for regression, not classification.
    # For classification, we use metrics like accuracy, precision, recall, and F1-score.
    
    report = classification_report(y_true, y_pred, labels=labels, output_dict=True, zero_division=0)
    
    # Convert confusion matrix to a list of lists for JSON serialization
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_list = cm.tolist()

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "classification_report": report,
        "confusion_matrix": {
            "labels": labels,
            "matrix": cm_list
        }
    }
    return metrics

def run_performance_test():
    """
    Runs a performance test on the sentiment and bot detection models
    using a ground truth dataset.
    """
    print("Starting performance test...")

    # 1. Load ground truth data
    ground_truth_data = load_ground_truth(GROUND_TRUTH_PATH)
    if not ground_truth_data:
        return

    # Create a DataFrame for easier manipulation
    df_truth = pd.DataFrame(ground_truth_data)
    comments_to_test = df_truth["Comment"].tolist()

    # 2. Prepare data for analysis functions
    # The analysis functions expect a simple list of strings in a JSON file.
    with open(TEMP_COMMENTS_PATH, "w", encoding="utf-8") as f:
        json.dump(comments_to_test, f, ensure_ascii=False, indent=4)

    # 3. Run the models to get predictions
    print("Running sentiment analysis model...")
    analyze_sentiment(TEMP_COMMENTS_PATH, TEMP_SENTIMENT_PATH)#,use_original_tags=False) #not every case needed the tuse original tags.
    
    print("Running bot detection model...")
    run_bot_detection(TEMP_COMMENTS_PATH, TEMP_BOT_PATH)

    # 4. Load prediction results
    with open(TEMP_SENTIMENT_PATH, "r", encoding="utf-8") as f:
        sentiment_predictions = json.load(f)
    
    with open(TEMP_BOT_PATH, "r", encoding="utf-8") as f:
        bot_predictions = json.load(f)

    # 5. Merge predictions with ground truth
    df_sentiment_preds = pd.DataFrame(sentiment_predictions)
    df_bot_preds = pd.DataFrame(bot_predictions)

    # Merge based on the 'Comment' column
    df_merged = pd.merge(df_truth, df_sentiment_preds, on="Comment", suffixes=('_true', '_pred'))
    df_merged = pd.merge(df_merged, df_bot_preds, on="Comment", suffixes=('_true', '_pred'))

    # 6. Calculate performance metrics
    
    # --- Sentiment Analysis Performance ---
    y_true_sentiment = df_merged['Sentiment_true']
    y_pred_sentiment = df_merged['Sentiment_pred']
    sentiment_labels = ["Positive", "Neutral", "Negative"] # Define order for confusion matrix
    
    print("\nCalculating sentiment analysis metrics...")
    sentiment_metrics = calculate_metrics(y_true_sentiment, y_pred_sentiment, sentiment_labels)

    # --- Bot Detection Performance ---
    y_true_bot = df_merged['Prediction_true']
    y_pred_bot = df_merged['Prediction_pred']
    bot_labels = ["human", "bot"] # Define order for confusion matrix

    print("Calculating bot detection metrics...")
    bot_metrics = calculate_metrics(y_true_bot, y_pred_bot, bot_labels)

    # 7. Compile the final report
    final_report = {
        "sentiment_analysis_performance": sentiment_metrics,
        "bot_detection_performance": bot_metrics
    }

    # 8. Save the report to a JSON file
    with open(PERFORMANCE_REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(final_report, f, ensure_ascii=False, indent=4)
    
    print(f"\nPerformance test complete. Report saved to {PERFORMANCE_REPORT_PATH}")
    print("\n--- Performance Report Summary ---")
    print(json.dumps(final_report, indent=2))
    print("--------------------------------\n")

    # 9. Clean up temporary files
    os.remove(TEMP_COMMENTS_PATH)
    os.remove(TEMP_SENTIMENT_PATH)
    os.remove(TEMP_BOT_PATH)
    print("Temporary files cleaned up.")


if __name__ == "__main__":
    # Ensure the data directory exists
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
    
    # Create a sample ground truth file if it doesn't exist.
    # You should expand this file with your own labeled data for a meaningful test.
    if not os.path.exists(GROUND_TRUTH_PATH):
        print(f"Sample ground truth file not found. Creating a new one at {GROUND_TRUTH_PATH}")
        from data.sample_ground_truth import sample_data
        with open(GROUND_TRUTH_PATH, "w", encoding="utf-8") as f:
            json.dump(sample_data, f, ensure_ascii=False, indent=4)

    run_performance_test()