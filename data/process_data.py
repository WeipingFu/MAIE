import pandas as pd
import numpy as np
from code.utils import load_jsonl, save_jsonl
from code.agents.PlanAgent import UserPrompt


def append_messages_to_plan(data_path, save_path, for_train=True):
    messages_list = []
    data = load_jsonl(data_path)
    user_prompt = UserPrompt(mode='plan')
    for idx, item in enumerate(data):
        convs = item['conversations']
        if isinstance(convs, np.ndarray):
            convs = convs.tolist()
        content = user_prompt.generate_user_prompt(
            task=item['question'], 
            model_responses=item['model_response'], 
            eval_mode=item['eval_type'],
            original_evaluation_plan=None,
            feedback=None, 
            convs=convs,
            criteria_list=None
        )
        messages = [
            {
                'role':'system', 
                'content': 'You are the Planning Agent in a multi-agent system. Your function is to design a detailed, structured evaluation plan for the given instance input. If the input contains revision feedback, revise the existing plan accordingly.'
            },
            {
                'role': 'user',
                'content': content
            }
        ]
        if for_train:
            messages.append({'role': 'assistant', 'content': item['evaluation_plan']})
        messages_list.append({'messages': messages})
    save_jsonl(messages_list, save_path)
    print(f'Save {len(messages_list)} data to {save_path}')


data_path = ''
save_path = ''
append_messages_to_plan(data_path, save_path, for_train=False)