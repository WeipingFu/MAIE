import random
import copy
from ..api_request import completion_json
from pydantic import BaseModel, Field
from typing import List, Literal, Optional
from ..utils import read_text, load_jsonl, save_jsonl, clean_json
import numpy as np
from jinja2 import Template
from code.data_generator.generate_plan import PlannerResponse
from tqdm import tqdm
import json


class CriticResponse(BaseModel):
    """
    Structured response returned by the Critic Agent.
    """
    decision: Literal["accept", "revise"] = Field(description="The decision Critic Agent make.")
    issues: Optional[List[str]] = Field(default_factory=list, description="Optional list of issues of the evaluation plan. If there are no issues, then return a blank list.")
    suggestion: str = Field(description="if decision = revise, return suggestions for revision; if decision = accept, return a short reason for this decision.")


class CombinedResponse(BaseModel):
    corrupted_plan: PlannerResponse = Field(description='The corrupted evaluation plan')
    critic: CriticResponse = Field(description='The critic result of corrupted evaluation plan')


def get_conv_history(convs):
        history_str = ''
        if convs and len(convs) > 0:
            history_str = '[Conversation History Between User and Models]\n'
            for idx, conv in enumerate(convs):
                history_str += '[Turn {}]\n[User]\n{}\n[Model a]\n{}\n[Model b]\n{}\n\n'.format(str(idx+1), conv['user'], conv['a'], conv['b'])
        return history_str
    
def get_user_criteria(eval_mode, criteria_list=None):
    user_criteria = ''
    scoring_scale = 'binary preference' if eval_mode == 'pairwise' else '1-5'
    if criteria_list and len(criteria_list) > 0:
        user_criteria += '[User-defined Evaluation Criteria]\n'
        for i, item in enumerate(criteria_list):
            dimension_str = item.get('dimension','')+"-"+item.get('description')
            one = f"Dimension: {dimension_str}\nScoring Scale: {item.get('scoring_scale', scoring_scale)}\nHigh Score Indicator: {item.get('high_score_indicator','')}\nLow Score Indicator: {item.get('low_score_indicator','')}\n"
            user_criteria += one
    return user_criteria

def get_model_response(model_responses):
    mapping = {i: chr(ord('a') + i - 1) for i in range(1, 27)}
    if len(model_responses) == 1:
        model_response_str = model_responses[0]
    else:
        model_response_str = '\n'.join([f'[Response of Model {mapping[idx+1]}]\n{response}' for idx, response in enumerate(model_responses)])
    return model_response_str


def corrupt_plan(prompt_path, model, item, corruption_type, prt=False):
    user_prompt = read_text(prompt_path)
    system_prompt = 'You are an expert evaluator and evaluation-plan engineer. You are the authority responsible for manipulating the Gold Evaluation Plan. Your core duty is to apply a specified Corruption Type to the gold plan and generate both the Corrupted Evaluation Plan and its corresponding Critic Result.'
    convs = item['conversations']
    if isinstance(convs, np.ndarray):
        convs = convs.tolist()
    template_vars = {
        "history": get_conv_history(convs),
        "task_description": item['question'],
        "model_response": get_model_response(item['model_response']),
        "criteria": get_user_criteria(item['eval_type'], criteria_list=None),
        "evaluation_plan": item['evaluation_plan'],
        "corruption_type": corruption_type
    }
    template = Template(user_prompt)
    content = template.render(**template_vars)
    messages = [
        {'role': 'system', 'content': system_prompt},
        {'role': 'user', 'content': content}
    ]
    if prt:
        print('Messages:')
        print(messages)
    clean_json_str = ''
    # resp = completion(model, messages, max_try=3, prt=prt)
    resp = completion_json(model, messages, CombinedResponse, max_try=3, prt=prt)
    if type(resp) is str:
        resp = clean_json(resp)
    clean_json_str = resp.model_dump_json()
    if prt:
        print('Response:')
        print(clean_json_str)
    critic_result, bad_plan = json.loads(clean_json_str)['critic'], json.loads(clean_json_str)['corrupted_plan']
    return critic_result, bad_plan


def build_critic_data_from_plan_gold(prompt_path, plan_gold_list, save_path, n_pos=800, n_neg=1200, prt=False):
    """
    plan_gold_list: list of dicts with keys ["task", "model_response", "evaluation_plan", ...]
    n_pos: number of positive (accept) samples
    n_neg: number of negative (revise) samples
    """
    N = len(plan_gold_list)
    indices = list(range(N))
    random.seed(42)
    random.shuffle(indices)

    # 1. select positive samples
    pos_indices = indices[:n_pos]
    pos_samples = []
    for idx in pos_indices:
        item = plan_gold_list[idx]
        item['corrupted_plan'] = None
        item['corruption_type'] = None
        item['critic_result'] = {
            "decision": "accept",
            "issues": [],
            "suggestion": "The evaluation plan is well-aligned with the task, complete, and executable for this instance."
        }
        pos_samples.append(item)

    # 2. sample plans, and corrupt them using GPT-4o
    neg_samples = []
    neg_base_indices = random.choices(indices, k=n_neg)
    # neg_base_indices = [neg_base_indices[0]]
    corruption_types = ['Alignment', 'Coverage', 'Granularity', 'Overlap', 'Measurability', 'Executability', 'Consistency']
    for idx in tqdm(neg_base_indices):
        item = plan_gold_list[idx]
        new_item = copy.deepcopy(item)
        corruption_type = random.choice(corruption_types)
        critic_result, bad_plan = corrupt_plan(prompt_path, 'gpt-4o', new_item, corruption_type, prt=prt)
        new_item['corrupted_plan'] = bad_plan
        new_item['corruption_type'] = corruption_type
        new_item['critic_result'] = critic_result
        neg_samples.append(new_item)
        if len(neg_samples) > 0 and len(neg_samples) % 2 == 0:
            dataset = pos_samples + neg_samples
            save_jsonl(dataset, save_path)

    dataset = pos_samples + neg_samples
    random.shuffle(dataset)
    save_jsonl(dataset, save_path)
    print(f'Save {len(dataset)} to {save_path}')
    return dataset


def get_critic_prompt(prompt_path, item):
    user_prompt = read_text(prompt_path)
    convs = item['conversations']
    if isinstance(convs, np.ndarray):
        convs = convs.tolist()
    evaluation_plan = item['evaluation_plan']
    if 'corrupted_plan' in item and item['corrupted_plan']:
        if type(item['corrupted_plan']) is dict:
            evaluation_plan = json.dumps(item['corrupted_plan'])
        else:
            evaluation_plan = item['corrupted_plan']
    template_vars = {
        "examples": '',
        "history": get_conv_history(convs),
        "task_description": item['question'],
        "model_response": get_model_response(item['model_response']),
        "criteria": get_user_criteria(item['eval_type'], criteria_list=None),
        "evaluation_plan": evaluation_plan
    }
    template = Template(user_prompt)
    content = template.render(**template_vars)
    return content


def build_critic_data_with_gpt(prompt_path, data_list, save_path, prt=False):
    results = []
    for idx, item in tqdm(enumerate(data_list), total=len(data_list)):
        content = get_critic_prompt(prompt_path, item)
        messages = [
            {'role': 'system', 'content': 'You are the Plan Critic. Your responsibility is to conduct a thorough review of the provided Evaluation Plan and give feedback.'},
            {'role': 'user', 'content': content}
        ]
        if prt:
            print('Messages:')
            print(messages)
        clean_json_str = ''
        resp = completion_json('gpt-4o', messages, CriticResponse, max_try=3, prt=prt)
        if type(resp) is str:
            resp = clean_json(resp)
        clean_json_str = resp.model_dump_json()
        if prt:
            print('Response:')
            print(clean_json_str)
        item['corrupted_plan'] = None
        item['corruption_type'] = None
        item['critic_result'] = clean_json_str
        results.append(item)
        if len(results) > 0 and len(results) % 2 == 0:
            save_jsonl(results, save_path)
    
    save_jsonl(results, save_path)
    print(f'Save {len(results)} data to {save_path}')


def append_critic_messages(prompt_path, data_list, save_path, for_train=True, prt=False):
    results = []
    random.shuffle(data_list)
    for idx, item in tqdm(enumerate(data_list), total=len(data_list)):
        content = get_critic_prompt(prompt_path, item)
        messages = [
            {'role': 'system', 'content': 'You are the Plan Critic. Your responsibility is to conduct a thorough review of the provided Evaluation Plan and give feedback.'},
            {'role': 'user', 'content': content}
        ]
        if for_train:
            messages.append({'role': 'assistant', 'content': item['critic_result']})
        if prt:
            print('Messages:')
            print(messages)
        results.append({'messages': messages})
    save_jsonl(results, save_path)
    print(f'Save {len(results)} to {save_path}')
        


if __name__ == "__main__":
    # # Corrupt gold plan
    # prompt_path = 'data_generator/corrupt_plan.txt'
    # plan_gold_list = load_jsonl('/Users/fuweiping/个人空间/DR/工作站/llmeval/adaptive/data/plan_train_sft.jsonl')
    # save_path = '/Users/fuweiping/个人空间/DR/工作站/llmeval/adaptive/data/critic_train_sft.jsonl'
    # build_critic_data_from_plan_gold(
    #     prompt_path, 
    #     plan_gold_list, 
    #     save_path, 
    #     n_pos=800, 
    #     n_neg=1200,
    #     prt=False
    # )


    # # Generate critic with gpt
    # prompt_path = 'prompts/critic.txt'
    # data_list = load_jsonl('/Users/fuweiping/个人空间/DR/工作站/llmeval/adaptive/data/planner_result.jsonl')
    # save_path = '/Users/fuweiping/个人空间/DR/工作站/llmeval/adaptive/data/critic_train_sft-2.jsonl'
    # # data_list = [data_list[0]]
    # build_critic_data_with_gpt(
    #     prompt_path, 
    #     data_list, 
    #     save_path, 
    #     prt=False
    # )

    # append messages for train
    data_list = load_jsonl('/Users/fuweiping/个人空间/DR/工作站/llmeval/adaptive/data/critic_train_sft.jsonl')
    save_path = '/Users/fuweiping/个人空间/DR/工作站/llmeval/adaptive/data/critic_sft.jsonl'
    append_critic_messages(
        prompt_path='prompts/critic.txt', 
        data_list=data_list, 
        save_path=save_path, 
        for_train=True, 
        prt=True
    )
