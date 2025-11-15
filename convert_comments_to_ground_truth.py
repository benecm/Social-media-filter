"""
Script to convert comments.json into ground_truth.json format.
Allows manual annotation of comments for dataset creation.
Saves progress incrementally to avoid data loss.
"""

import json
import os
from pathlib import Path


def load_comments(filepath):
    """Load comments from JSON file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_existing_ground_truth(filepath):
    """Load existing ground truth data if it exists."""
    if filepath.exists():
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    return []


def append_to_ground_truth(entry, filepath):
    """Append a single entry to the ground truth JSON file."""
    data = load_existing_ground_truth(filepath)
    data.append(entry)
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


def get_sentiment_input():
    """Get sentiment classification from user using numeric shortcuts."""
    while True:
        choice = input("  Sentiment - (1) Positive, (2) Negative, (3) Neutral: ").strip()
        if choice == '1':
            return "Positive"
        elif choice == '2':
            return "Negative"
        elif choice == '3':
            return "Neutral"
        print("  Invalid input. Please enter 1, 2, or 3")


def get_prediction_input():
    """Get bot/human prediction from user using numeric shortcuts."""
    while True:
        choice = input("  Prediction - (1) Human, (2) Bot: ").strip()
        if choice == '1':
            return "human"
        elif choice == '2':
            return "bot"
        print("  Invalid input. Please enter 1 or 2")


def annotate_comments(comments, ground_truth_file, already_annotated):
    """Annotate comments with sentiment and bot prediction."""
    total_comments = len(comments)
    
    print(f"\nAnnotating {total_comments} comments...")
    print(f"Already annotated: {already_annotated}")
    print(f"Remaining: {total_comments - already_annotated}\n")
    
    for i, comment in enumerate(comments, 1):
        # Skip already annotated comments
        if i <= already_annotated:
            continue
        
        print(f"[{i}/{total_comments}]")
        print(f"Comment: {comment}")
        
        sentiment = get_sentiment_input()
        prediction = get_prediction_input()
        
        entry = {
            "Comment": comment,
            "Sentiment": sentiment,
            "Prediction": prediction
        }
        
        # Append to file immediately
        append_to_ground_truth(entry, ground_truth_file)
        print("✓ Saved\n")


def main():
    """Main function."""
    # Define file paths
    script_dir = Path(__file__).parent
    comments_file = script_dir / "data" / "comments.json"
    ground_truth_file = script_dir / "data" / "ground_truth_new.json"
    
    # Check if comments file exists
    if not comments_file.exists():
        print(f"Error: {comments_file} not found")
        return
    
    # Load comments
    comments = load_comments(comments_file)
    print(f"Loaded {len(comments)} comments from {comments_file}")
    
    # Check for existing annotations
    existing_data = load_existing_ground_truth(ground_truth_file)
    already_annotated = len(existing_data)
    
    if already_annotated > 0:
        print(f"Found {already_annotated} previously annotated comments in {ground_truth_file}")
    
    # Annotate comments
    annotate_comments(comments, ground_truth_file, already_annotated)
    
    # Load final data to show summary
    final_data = load_existing_ground_truth(ground_truth_file)
    print(f"\n✓ Dataset complete!")
    print(f"Total annotations: {len(final_data)}")
    print(f"Output file: {ground_truth_file}")


if __name__ == "__main__":
    main()
