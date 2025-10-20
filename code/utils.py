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