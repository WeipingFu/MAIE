import json
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def _abs_path(path: str) -> str:
    if os.path.isabs(path):
        return path
    return os.path.join(BASE_DIR, path)

# -----------------------
# TEXT
# -----------------------
def read_text(path: str) -> str:
    abs_path = _abs_path(path)
    with open(abs_path, 'r', encoding='utf-8') as f:
        content = f.read()
    return content

def save_text(s: str, path: str):
    abs_path = _abs_path(path)
    with open(abs_path, 'w', encoding='utf-8') as f:
        f.write(s)

# -----------------------
# JSON
# -----------------------
def load_json(file_path: str):
    abs_path = _abs_path(file_path)
    with open(abs_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def save_json(data, save_path: str):
    abs_path = _abs_path(save_path)
    with open(abs_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

# -----------------------
# JSONL
# -----------------------
def load_jsonl(json_path: str):
    abs_path = _abs_path(json_path)
    with open(abs_path, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f if line.strip()]
    return data

def save_jsonl(data_list, file_path: str, ensure_ascii=False):
    abs_path = _abs_path(file_path)
    jsonl_content = (json.dumps(item, ensure_ascii=ensure_ascii) + '\n' 
                     for item in data_list)
    with open(abs_path, 'w', encoding='utf-8') as f:
        f.writelines(jsonl_content)


def safe_load_json(content):
    if isinstance(content, str):
        content = content.strip().replace('```json','').replace('```','')
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            return {}
    elif isinstance(content, dict):
        return content
    else:
        return {}