import pandas as pd
from sklearn.metrics import classification_report, accuracy_score



def compute_metrics_pariwise(df, label_col, pred_col):
    # map prediction with ground_truth
    df = df.dropna()
    gt_map = {
        "model_a": "model_a",
        "model_b": "model_b",
        "tie": "tie"
    }
    df["gt_label"] = (
        df[label_col]
        .astype(str)
        .str.strip()
        .str.lower()
        .map(gt_map)
    )
    df["pred_label"] = (
        df[pred_col]
        .astype(str)
        .str.strip()
        .str.lower()
    )
    y_true = df["gt_label"]
    y_pred = df["pred_label"]
    print("Accuracy:", round(accuracy_score(y_true, y_pred), 3))
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, digits=3))


if __name__ == "__main__":
    df = pd.read_excel("/Users/fuweiping/个人空间/DR/工作站/llmeval/adaptive/result/mt-bench/vanilla_cot/vanilla-qwen3-4b.xlsx")
    label_col = 'winner'
    pred_col = 'judgement'
    compute_metrics_pariwise(df, label_col, pred_col)
