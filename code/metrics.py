import pandas as pd
from sklearn.metrics import classification_report, accuracy_score
from utils import load_jsonl, save_jsonl
from scipy.stats import pearsonr


def compute_metrics_pariwise(df, label_col, pred_col):
    # map prediction with ground_truth
    # df = df.dropna(subset=['judgement'])
    # df = df[df['judgement']!='tie']
    print(len(df))
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


def compute_metrics_pointwise(df, label_col, pred_col):
    # notdf = df[(df[pred_col] < 1) | (df[pred_col] > 5) | (df[pred_col].isna())]
    # print(notdf)
    # df = df.dropna(subset=[pred_col])
    df[label_col] = pd.to_numeric(df[label_col], errors='coerce')
    # df[label_col] = df[label_col].round(2)
    df[pred_col] = pd.to_numeric(df[pred_col], errors='coerce')
    # df[pred_col] = df[pred_col].round(2)
    df = df[(df[pred_col] >= 1) & (df[pred_col] <= 5)]
    print(f"data count: {df.shape[0]}")
    label_data = df[label_col].to_list()
    pred_data = df[pred_col].to_list()
    pearson_r, p_value = pearsonr(label_data, pred_data)
    pearson = round(pearson_r, 3)
    print(f'Pearson Correlation: {pearson}, p value: {p_value}')




if __name__ == "__main__":
    # df = pd.read_excel("/Users/fuweiping/个人空间/DR/工作站/llmeval/adaptive/result/mt-bench/ablation/plan-critic-judge-chat-qwen3-8b-old.xlsx")
    df = pd.DataFrame(load_jsonl('/Users/fuweiping/个人空间/DR/工作站/llmeval/adaptive/result/judgebench/ablation/plan-judge-chat-qwen3-8b.jsonl'))
    print(len(df[df['judgement'].isna()]), len(df[df['judgement']=='tie']))
    label_col = 'winner'
    pred_col = 'judgement'
    compute_metrics_pariwise(df, label_col, pred_col)

    # # df = pd.read_excel('/Users/fuweiping/个人空间/DR/工作站/llmeval/adaptive/result/flask/plan-critic-judge-chat-qwen3-8b-criteria.xlsx')
    # df = pd.DataFrame(load_jsonl("/Users/fuweiping/个人空间/DR/工作站/llmeval/adaptive/result/feedbackbench/plan-judge-qwen3-8b-criteria.jsonl"))
    # # df.to_excel('/Users/fuweiping/个人空间/DR/工作站/llmeval/adaptive/result/rewardbench2/vanilla/vanilla-llama-3.1-8b-instruct.xlsx', index=False)
    # print(len(df[df['judgement'].isna()]))
    # label_col = 'score'
    # pred_col = 'judgement'
    # compute_metrics_pointwise(df, label_col, pred_col)

    # data = load_jsonl("/Users/fuweiping/个人空间/DR/工作站/llmeval/adaptive/result/rewardbench2/plan-critic-judge-chat-qwen3-8b.jsonl")
    # temp = load_jsonl("/Users/fuweiping/个人空间/DR/工作站/llmeval/adaptive/result/rewardbench2/plan-critic-judge-chat-qwen3-8b-1.jsonl")
    # new_data = []
    # for item in data:
    #     if not item['judgement'] or item['judgement']=='tie':
    #         one = [x for x in temp if x['id']==item['id'] and x['question']==item['question']]
    #         if len(one) > 0:
    #             new_data.append(one[0])
    #         else:
    #             new_data.append(item)
    #     else:
    #         new_data.append(item)

    # print(len(new_data))
    # save_jsonl(new_data, '/Users/fuweiping/个人空间/DR/工作站/llmeval/adaptive/result/rewardbench2/plan-critic-judge-chat-qwen3-8b-new.jsonl')
    