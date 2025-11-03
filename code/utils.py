import json

def read_text(path): 
    with open(path, 'r', encoding='utf-8') as f: 
        content = f.read()
    return str(content)

def save_text(s, path):
    with open(path, 'w', encoding='utf-8') as f:
        f.write(s)
        
def load_json(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

def save_json(data, save_path):
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False)

def load_jsonl(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f if line.strip()]
    return data

def save_jsonl(data_list, file_path, ensure_ascii=False):
    jsonl_content = (json.dumps(item, ensure_ascii=ensure_ascii) + '\n' 
                     for item in data_list)
    with open(file_path, 'w', encoding='utf-8') as f:
        f.writelines(jsonl_content)