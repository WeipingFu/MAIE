import json
import pandas as pd
import re
from utils import load_jsonl, save_jsonl

def extract_vanilla_pairwise(text):
    if text is None or (isinstance(text, float) and pd.isna(text)):
        return None

    if isinstance(text, dict):
        return text.get("judgement")

    text = str(text).strip()
    json_blocks = re.findall(r'\{[\s\S]*?\}', text)
    if json_blocks:
        last_json = json_blocks[-1]
        try:
            data = json.loads(last_json)
            if isinstance(data, dict) and "judgement" in data:
                return data["judgement"]
        except Exception:
            pass  

    pattern = re.compile(
        r'["\']?judgement["\']?\s*[:：]\s*["\']?(model_a|model_b|tie)["\']?',
        re.IGNORECASE
    )
    match = pattern.search(text)
    if match:
        return match.group(1).lower()

    lower_text = text.lower()
    has_a = "model_a" in lower_text
    has_b = "model_b" in lower_text
    if has_a and not has_b:
        return "model_a"
    if has_b and not has_a:
        return "model_b"

    return None

def extract_vanilla_pointwise(text):
    if text is None or (isinstance(text, float) and pd.isna(text)):
        return None

    if isinstance(text, dict):
        return text.get("judgement")
    
    text = str(text).strip()
    json_blocks = re.findall(r'\{[\s\S]*?\}', text)
    if json_blocks:
        last_json = json_blocks[-1]
        try:
            data = json.loads(last_json)
            if isinstance(data, dict) and "judgement" in data:
                return data["judgement"]
        except Exception:
            pass
    return ''

def handle_vanilla(result_path, result_col, mode='pairwise', save_path=None):
    if '.xlsx' in result_path:
        res = pd.read_excel(result_path).to_dict(orient='records')
    elif '.jsonl' in result_path:
        res = load_jsonl(result_path)
    print(f'data count = {len(res)}')

    new_data = []
    for idx, row in enumerate(res):
        if mode == 'pairwise':
            judgement = extract_vanilla_pairwise(row[result_col])
        else:
            judgement = extract_vanilla_pointwise(row[result_col])
        row['judgement'] = judgement
        new_data.append(row)

    # save results
    if save_path and '.xlsx' in save_path:
        pd.DataFrame(new_data).to_excel(save_path, index=False)
        print(f"Result saved to {save_path}")
    elif save_path and '.jsonl' in save_path:
        save_jsonl(new_data, save_path)
        print(f"Result saved to {save_path}")


if __name__ == "__main__":
    result_path = '/Users/fuweiping/个人空间/DR/工作站/llmeval/adaptive/result/rewardbench2/vanilla/vanilla-llama-3.1-8b-instruct.xlsx'
    result_col = 'vanilla_prompt'
    save_path = result_path.replace('.jsonl', '.xlsx')
    handle_vanilla(result_path, result_col, mode='pairwise', save_path=save_path)