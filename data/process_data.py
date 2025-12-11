from ..code.utils import load_jsonl, save_jsonl


def process_plan_sft_data(data_path, save_path):
    messages_list = []
    data = load_jsonl(data_path)
    for idx, item in enumerate(data):
        messages = item['messages']
        messages_list.append({'messages': messages})
    save_jsonl(messages_list, save_path)
    print(f'Save {len(messages_list)} data to {save_path}')

data_path = ''
save_path = ''
process_plan_sft_data(data_path, save_path)