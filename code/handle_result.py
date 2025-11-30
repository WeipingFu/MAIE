import json
import pandas as pd
import re


def extract_vanilla_judgement(text):
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
    result_path = '/Users/fuweiping/个人空间/DR/工作站/llmeval/adaptive/result/mt-bench/vanilla_cot/vanilla-qwen3-8b.xlsx'
    result_col = 'vanilla_prompt'
    save_path = result_path
    handle_vanilla_excel(result_path, result_col, save_path)