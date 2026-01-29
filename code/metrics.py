import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, accuracy_score
from utils import load_jsonl, save_jsonl, load_jsonl_safe
from scipy.stats import pearsonr
import json


def compute_metrics_pariwise(df, label_col, pred_col, with_tie=True):
    # map prediction with ground_truth
    df = df.dropna(subset=['judgement'])
    # df = df[df['judgement']!='tie']
    if not with_tie:
        df = df[df[label_col]!='tie']
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
    notdf = df[(df[pred_col] < 1.0) | (df[pred_col] > 5.1) | (df[pred_col].isna())]
    print(notdf)
    df = df.dropna(subset=[pred_col])
    df = df[(df[pred_col] >= 1.0) & (df[pred_col] <= 5.1)]
    df[label_col] = pd.to_numeric(df[label_col], errors='coerce')
    df[pred_col] = pd.to_numeric(df[pred_col], errors='coerce')
    # df = df[(df[pred_col] >= 1) & (df[pred_col] <= 5)]

    # Pearson
    print(f"data count: {df.shape[0]}")
    label_data = df[label_col].to_list()
    pred_data = df[pred_col].to_list()
    pearson_r, p_value = pearsonr(label_data, pred_data)
    pearson = round(pearson_r, 3)
    print(f'Pearson Correlation: {pearson}, p value: {p_value}')

    # MAE
    mae, diff, used_pred = mean_absolute_error_adaptive(
        label_data, pred_data,
        gt_min=1.0, gt_max=5.0,
        # pred_min=1.0, pred_max=5.0
    )
    print(f'MAE: {mae}')


def mean_absolute_error_adaptive(
    groundtruth,
    prediction,
    gt_min=None,
    gt_max=None,
    pred_min=None,
    pred_max=None
):
    """
    Adaptive MAE:
    - If gt/pred ranges are the same -> directly compute MAE
    - If ranges differ -> normalize prediction to gt range, then compute MAE
    """

    gt = np.asarray(groundtruth, dtype=float)
    pred = np.asarray(prediction, dtype=float)

    assert gt.shape == pred.shape, "groundtruth and prediction must have same length"

    # infer ranges if not provided
    if gt_min is None: gt_min = gt.min()
    if gt_max is None: gt_max = gt.max()
    if pred_min is None: pred_min = pred.min()
    if pred_max is None: pred_max = pred.max()

    # same scale -> no normalization
    if (gt_min, gt_max) == (pred_min, pred_max):
        abs_diff = np.abs(gt - pred)
        return abs_diff.mean(), abs_diff, pred

    # different scale -> normalize prediction
    if (gt_min, gt_max) != (pred_min, pred_max):
        pred = np.clip(pred, pred_min, pred_max)
        print('prediction normalized!')

    pred_norm = (
        (pred - pred_min) / (pred_max - pred_min)
        * (gt_max - gt_min)
        + gt_min
    )

    abs_diff = np.abs(gt - pred_norm)
    return abs_diff.mean(), abs_diff, pred_norm


def load_multi_line_json_items(path):
        with open(path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        results = []
        decoder = json.JSONDecoder()
        pos = 0
        content = content.strip()
        
        while pos < len(content):
            try:
                # 自动跳过空白符，并解析出从当前位置开始的第一个完整 JSON 对象
                obj, pos = decoder.raw_decode(content, pos)
                results.append(obj)
                
                # 跳过对象之间的换行符或空格
                while pos < len(content) and content[pos].isspace():
                    pos += 1
            except json.JSONDecodeError as e:
                # 如果中间有脏数据，打印并停止
                print(f"解析到字符位置 {pos} 时出错: {e}")
                break
                
        return results




if __name__ == "__main__":
    # # df = pd.read_excel("/Users/fuweiping/个人空间/DR/工作站/llmeval/adaptive/result/mt-bench/plan-critic-judge-all-chat-qwen3-8b.xlsx")
    # df = pd.DataFrame(load_jsonl('/Users/fuweiping/个人空间/DR/工作站/llmeval/adaptive/result/rewardbench/plan-critic-judge-all-chat-qwen3-8b.jsonl'))
    # # df = pd.DataFrame(load_multi_line_json_items('/Users/fuweiping/个人空间/DR/工作站/llmeval/adaptive/result/mt-bench/praetor-mt-bench/Praetor-mt-bench.jsonl'))
    # print(len(df[df['judgement'].isna()]), len(df[df['judgement']=='tie']))
    # label_col = 'winner'
    # pred_col = 'judgement'
    # compute_metrics_pariwise(df, label_col, pred_col, with_tie=False)
    

    
    # df = pd.read_excel('/Users/fuweiping/个人空间/DR/工作站/llmeval/adaptive/result/flask/vanilla/vanilla-qwen3-8b-criteria.xlsx')
    # # df = pd.DataFrame(load_jsonl("/Users/fuweiping/个人空间/DR/工作站/llmeval/adaptive/result/flask/plan-critic-judge-all-chat-qwen3-8b-criteria.jsonl"))
    # # df.to_excel('/Users/fuweiping/个人空间/DR/工作站/llmeval/adaptive/result/flask_sum/vanilla-qwen3-8b.xlsx', index=False)
    # label_col = 'avg_human_score'
    # pred_col = 'judgement'
    # compute_metrics_pointwise(df, label_col, pred_col)


    # data = load_jsonl("/Users/fuweiping/个人空间/DR/工作站/llmeval/adaptive/result/flask/plan-critic-judge-chat-qwen3-8b-criteria.jsonl")
    # temp = load_jsonl('/Users/fuweiping/个人空间/DR/工作站/llmeval/adaptive/result/flask/plan-critic-judge-chat-qwen3-8b-criteria-1.jsonl')
    # new_data = []
    # for item in data:
    #     if not item['judgement'] or float(item['judgement'])<1 or float(item['judgement'])>5:
    #         one = [x for x in temp if x['task_instruction']==item['task_instruction'] and x['model_response']==item['model_response'] and x['dimension']==item['dimension']]
    #         if len(one) > 0:
    #             new_data.append(one[0])
    #     else:
    #         new_data.append(item)
    
    # print(len(new_data))
    # save_jsonl(new_data, '/Users/fuweiping/个人空间/DR/工作站/llmeval/adaptive/result/flask/plan-critic-judge-chat-qwen3-8b-criteria-new.jsonl')

    data = load_jsonl('/Users/fuweiping/Downloads/train.jsonl')
    language = ['php', 'c', 'bash', 'sql', 'python', 'go', 'javascript_html_css', 'english', 'typescript', 'c#', 'java', 'powershell', 'r', 'c++']
    data = [x for x in data if x['language'] in language]
    print(len(data))
    print(len(set([x['context'][-1]['content'] for x in data])))
    