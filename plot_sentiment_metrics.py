import json
from pathlib import Path
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def load_metrics(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def plot_accuracy_bar(metrics: dict, out_path: Path):
    names = []
    accuracies = []
    for model_name, info in metrics.items():
        names.append(model_name)
        accuracies.append(info.get("accuracy", float("nan")))

    fig, ax = plt.subplots(figsize=(max(6, len(names) * 0.9), 4))
    sns.barplot(x=accuracies, y=names, palette="muted", ax=ax)
    ax.set_xlabel("Accuracy")
    ax.set_title("Model Accuracy Comparison")
    ax.set_xlim(0, 1)
    for i, v in enumerate(accuracies):
        ax.text(v + 0.01, i, f"{v:.2f}", va="center")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_confusion_matrices(metrics: dict, out_path: Path, cmap: str = "Blues"):
    items = list(metrics.items())
    n = len(items)
    cols = 3
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3))
    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = np.array([axes])

    axes = axes.reshape(rows, cols)

    for idx, (model_name, info) in enumerate(items):
        r = idx // cols
        c = idx % cols
        ax = axes[r, c]

        cm_info = info.get("confusion_matrix")
        if not cm_info:
            ax.text(0.5, 0.5, "No confusion matrix", ha="center", va="center")
            ax.set_axis_off()
            continue

        labels = list(cm_info.get("labels", []))
        matrix = np.array(cm_info.get("matrix", []))

        sns.heatmap(matrix, annot=True, fmt="d", cmap=cmap, xticklabels=labels, yticklabels=labels, ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title(model_name)

    for idx in range(n, rows * cols):
        r = idx // cols
        c = idx % cols
        axes[r, c].set_axis_off()

    fig.suptitle("Confusion Matrices", fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Plot metrics comparison charts")
    parser.add_argument("--metrics", "-m", default="data/sentiment_metrics.json", help="Path to metrics JSON")
    parser.add_argument("--outdir", "-o", default="plots", help="Output directory for charts")
    parser.add_argument("--show", action="store_true", help="Show plots interactively (will block)")
    args = parser.parse_args()

    metrics_path = Path(args.metrics)
    outdir = Path(args.outdir)

    if not metrics_path.exists():
        raise FileNotFoundError(f"Metrics file not found: {metrics_path}")

    metrics = load_metrics(metrics_path)

    acc_out = outdir / "accuracy_comparison.png"
    cm_out = outdir / "confusion_matrices.png"

    plot_accuracy_bar(metrics, acc_out)
    plot_confusion_matrices(metrics, cm_out)

    print(f"Saved accuracy chart to: {acc_out}")
    print(f"Saved confusion matrices to: {cm_out}")

    if args.show:
        try:
            import PIL.Image as Image

            Image.open(acc_out).show()
            Image.open(cm_out).show()
        except Exception:
            print("Unable to open images for display; they were saved to disk.")


if __name__ == "__main__":
    main()
