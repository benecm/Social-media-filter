import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def from_classification_report_to_df(cr: dict) -> pd.DataFrame:
    # Filter out summary rows like 'accuracy', 'macro avg', 'weighted avg'
    rows = {k: v for k, v in cr.items() if k not in ("accuracy", "macro avg", "weighted avg")}
    df = pd.DataFrame(rows).T
    # classification_report fields might be floats - ensure numeric
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    return df


def plot_confusion_matrix(matrix, labels, title, outpath: Path, normalize=False):
    cm = np.array(matrix)
    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True)
        with np.errstate(all='ignore'):
            cm_norm = cm / row_sums
        data = cm_norm
        fmt = ".2f"
        suffix = "_normalized"
    else:
        data = cm
        fmt = "d"
        suffix = ""

    plt.figure(figsize=(6, 5))
    sns.heatmap(data, annot=True, fmt=fmt, cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(title + (" (normalized)" if normalize else ""))
    plt.tight_layout()
    outfile = outpath / f"confusion_matrix{suffix}.png"
    plt.savefig(outfile)
    plt.close()
    return outfile


def plot_metrics_bars(df: pd.DataFrame, title: str, outpath: Path):
    # df expected to have precision, recall, f1-score and support (support may be int)
    metrics = ['precision', 'recall', 'f1-score']
    fig, ax = plt.subplots(figsize=(8, 4))
    df_metrics = df[metrics]
    df_metrics.plot(kind='bar', ax=ax)
    ax.set_ylim(0, 1)
    ax.set_title(title)
    ax.legend(title='Metric')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    outfile = outpath / "metrics_precision_recall_f1.png"
    plt.savefig(outfile)
    plt.close()
    return outfile


def plot_support_bar(df: pd.DataFrame, title: str, outpath: Path):
    if 'support' not in df.columns:
        return None
    fig, ax = plt.subplots(figsize=(6, 3))
    df['support'].astype(int).plot(kind='bar', ax=ax, color='grey')
    ax.set_title(title + ' - support')
    ax.set_ylabel('Support (count)')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    outfile = outpath / "support.png"
    plt.savefig(outfile)
    plt.close()
    return outfile


def save_classification_table(df: pd.DataFrame, title: str, outpath: Path):
    # Save a table representation as an image via matplotlib table
    fig, ax = plt.subplots(figsize=(6, 0.6 + 0.4 * len(df)))
    ax.axis('off')
    ax.set_title(title)
    tbl = ax.table(cellText=np.round(df.fillna(''), 3).values,
                   colLabels=df.columns,
                   rowLabels=df.index,
                   loc='center')
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1, 1.2)
    plt.tight_layout()
    outfile = outpath / "classification_report_table.png"
    plt.savefig(outfile)
    plt.close()
    return outfile


def generate_index_html(image_paths, outpath: Path):
    lines = ["<html><head><meta charset='utf-8'><title>Performance Plots</title></head><body>"]
    lines.append("<h1>Performance Plots</h1>")
    for p in image_paths:
        rel = p.name
        lines.append(f"<div style='margin:10px 0'><h3>{rel}</h3><img src='{rel}' style='max-width:100%;height:auto'></div>")
    lines.append("</body></html>")
    out_file = outpath / 'index.html'
    out_file.write_text('\n'.join(lines), encoding='utf-8')
    return out_file


def visualize_task(name: str, perf: dict, outdir: Path):
    task_dir = outdir / name
    ensure_dir(task_dir)
    created = []

    # Confusion matrices
    cm_info = perf.get('confusion_matrix', {})
    labels = cm_info.get('labels', [])
    matrix = cm_info.get('matrix', [])
    if labels and matrix:
        created.append(plot_confusion_matrix(matrix, labels, f"{name} Confusion Matrix", task_dir, normalize=False))
        created.append(plot_confusion_matrix(matrix, labels, f"{name} Confusion Matrix", task_dir, normalize=True))

    # Classification report
    cr = perf.get('classification_report', {})
    if cr:
        df = from_classification_report_to_df(cr)
        # drop possible non-numeric rows
        created.append(plot_metrics_bars(df, f"{name} - Precision/Recall/F1 by class", task_dir))
        supp = plot_support_bar(df, name, task_dir)
        if supp:
            created.append(supp)
        created.append(save_classification_table(df, f"{name} - classification report", task_dir))

    # Accuracy summary text image
    acc = perf.get('accuracy')
    if acc is not None:
        fig, ax = plt.subplots(figsize=(4, 1.2))
        ax.axis('off')
        ax.text(0.5, 0.5, f"Accuracy: {acc:.2f}", ha='center', va='center', fontsize=18)
        out = task_dir / 'accuracy.png'
        plt.savefig(out, bbox_inches='tight')
        plt.close()
        created.append(out)

    # return list of created files
    return [p for p in created if p is not None]


def main(json_path: Path, plots_root: Path):
    ensure_dir(plots_root)
    data = json.loads(json_path.read_text(encoding='utf-8'))
    created_all = []

    for key, perf in data.items():
        # friendly folder name
        folder = key.replace('_performance', '')
        created = visualize_task(folder, perf, plots_root)
        created_all.extend(created)

    # Accuracy comparison across tasks
    accs = {k: v.get('accuracy') for k, v in data.items()}
    if accs:
        fig, ax = plt.subplots(figsize=(6, 3))
        names = [k.replace('_performance', '') for k in accs.keys()]
        vals = [v for v in accs.values()]
        ax.bar(names, vals, color=['#4c72b0', '#55a868'][:len(names)])
        ax.set_ylim(0, 1)
        ax.set_ylabel('Accuracy')
        ax.set_title('Accuracy comparison')
        for i, v in enumerate(vals):
            ax.text(i, v + 0.01, f"{v:.2f}", ha='center')
        out = plots_root / 'accuracy_comparison.png'
        plt.tight_layout()
        plt.savefig(out)
        plt.close()
        created_all.append(out)

    # Create simple index.html inside plots_root and in each subfolder
    # Copy images (only names) into index for top-level
    top_images = [p for p in created_all if p.parent == plots_root]
    # Also include files from subfolders
    for sub in plots_root.iterdir():
        if sub.is_dir():
            imgs = list(sub.glob('*.png'))
            if imgs:
                # create index.html in subfolder
                generate_index_html(imgs, sub)

    # generate top-level index
    all_images_flat = [p for p in plots_root.rglob('*.png')]
    generate_index_html(all_images_flat, plots_root)

    print(f"Saved {len(all_images_flat)} images into {plots_root.resolve()}")
    print(f"Open {plots_root / 'index.html'} to view them.")


if __name__ == '__main__':
    repo_root = Path(__file__).parent
    json_path = repo_root / 'data' / 'performance_report.json'
    plots_root = repo_root / 'plots'
    if not json_path.exists():
        print(f"Could not find {json_path}. Run this script from the repo root.")
    else:
        main(json_path, plots_root)
