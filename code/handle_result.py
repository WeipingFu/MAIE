import json
import pandas as pd
import re


def extract_vanilla_judgement(json_str):
    if pd.isna(json_str):
        return None

    text = json_str.strip()
    cleaned = text
    if cleaned.startswith('"') and cleaned.endswith('"'):
        cleaned = cleaned[1:-1]

    cleaned = cleaned.replace('""', '"')

    try:
        data = json.loads(cleaned)
        return data.get("judgement", None)
    except Exception as e:
        pattern = r'"judgement"\s*:\s*"([^"]+)"'
        match = re.search(pattern, text)
    if match:
        return match.group(1)
    return None

def handle_vanilla_excel(result_path, result_col, save_path=None):
    resdf = pd.read_excel(result_path)
    judgements = []
    for idx, row in resdf.iterrows():
        judgement = extract_vanilla_judgement(row[result_col])
        judgements.append(judgement)
    resdf['judgement'] = judgements
    if save_path and '.xlsx' in save_path:
        resdf.to_excel(save_path, index=False)
        print(f"Result saved to {save_path}")


if __name__ == "__main__":
    result_path = '/Users/fuweiping/个人空间/DR/工作站/llmeval/adaptive/result/mt-bench/vanilla-llama3.1.xlsx'
    result_col = 'vanilla_prompt'
    save_path = result_path
    handle_vanilla_excel(result_path, result_col, save_path)