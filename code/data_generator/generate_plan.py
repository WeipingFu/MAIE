from ..utils import read_text, load_jsonl, save_jsonl, clean_json
import pandas as pd
from ..api_request import completion, completion_json
# from agents.PlanAgent import UserPrompt, PlannerResponse
from typing import List, Literal, Optional
from pydantic import BaseModel, Field
from jinja2 import Template
from tqdm import tqdm
import numpy as np
import json


# --- JudgeAgent Details ---
class AgentDetails(BaseModel):
    """Defines the role, task, and specific steps for the evaluation Agent."""
    role_name: str = Field(description="The name of the JudgeAgent.")
    role_description: str = Field(description="The role the Agent plays and its primary responsibilities.")
    evaluation_task: str = Field(description="A concise description of the evaluation task the Agent must judge.")
    evaluation_steps: List[str] = Field(description="Detailed, step-by-step guidelines that the Agent must follow.")

# --- Evaluation Dimension ---
class EvaluationDimension(BaseModel):
    """Defines a single dimension within a multi-dimensional evaluation system."""
    name: str = Field(description="The name of the dimension.")
    definition: str = Field(description="A definition describing what this dimension measures.")
    rationale: str = Field(description="Explanation of why this dimension is important for the evaluation.")
    weight: float = Field(description="The weight of this dimension in the final total score (0.0-1.0). The sum of all weights should be 1.")
    evaluation_granularity: str = Field(
        description='The scope and granularity of the evaluation: "holistic" (judges the response as a whole), "localized" (focuses on specific portions), "stepwise" (examines each reasoning or generation step).'
    )
    # evaluation_granularity: Literal["holistic", "localized", "stepwise"] = Field(
    #     description='The scope and granularity of the evaluation: "holistic" (judges the response as a whole), "localized" (focuses on specific portions), "stepwise" (examines each reasoning or generation step).'
    # )
    scoring_scale: str = Field(description='The scoring system used by the evaluator, e.g., "1-5", "0-100", or "binary preference".')
    high_score_indicator: str = Field(description="Describes the characteristics of a high score (full marks).")
    low_score_indicator: str = Field(description="Describes the characteristics of a low score (minimum marks).")
    dependencies: Optional[List[str]] = Field(default_factory=list, description="Optional list of names of other dimensions this dimension related with.")
    assigned_agent: AgentDetails = Field(description="The configuration for the JuegeAgent assigned to perform this dimension's evaluation.")

# --- Format of Planner Response ---
class PlannerResponse(BaseModel):
    """
    Structured response returned by the Planner Agent, used to guide subsequent evaluation flows.
    """
    task_type: str = Field(description="Identifies the type of task, e.g., 'Code Generation', 'Summarization'.")
    sub_task: str = Field(description="The specific sub-task.")
    evaluation_mode: Literal["pointwise", "pairwise"] = Field(
        description="Specifies whether the evaluation targets a single response (pointwise) or compares two responses (pairwise)."
    )
    evaluation_dimensions: List[EvaluationDimension] = Field(
        description="A list of multi-dimensional criteria to be used for evaluating the generation result."
    )

# --- User prompt for Planner ---
class UserPrompt:
    def __init__(self, prompt_path):
            self._user_prompt = read_text(prompt_path)
    def get_conv_history(self, convs):
        history_str = ''
        if convs and len(convs) > 0:
            history_str = '[Conversation History Between User and Models]\n'
            for idx, conv in enumerate(convs):
                history_str += '[Turn {}]\n[User]\n{}\n[Model a]\n{}\n[Model b]\n{}\n\n'.format(str(idx+1), conv['user'], conv['a'], conv['b'])
        return history_str
    
    def get_user_criteria(self, eval_mode, criteria_list=None):
        user_criteria = ''
        scoring_scale = 'binary preference' if eval_mode == 'pairwise' else '1-5'
        if criteria_list and len(criteria_list) > 0:
            # [{"dimension":"", "description":"", "scoring_scale":"", "high_score_indicator":"", "low_score_indicator":""}, ...]
            user_criteria += '[User-defined Evaluation Criteria]\n'
            for i, item in enumerate(criteria_list):
                dimension_str = item.get('dimension','')+"-"+item.get('description')
                one = f"Dimension: {dimension_str}\nScoring Scale: {item.get('scoring_scale', scoring_scale)}\nHigh Score Indicator: {item.get('high_score_indicator','')}\nLow Score Indicator: {item.get('low_score_indicator','')}\n"
                user_criteria += one
        return user_criteria
    
    def get_examplestr(self, example_paths=None):
        example_str = ''
        if example_paths:
            examples = []
            for path in example_paths:
                examples.append(read_text(path))
            example_str = '\n\n'.join(['[Start of Example {}]\n{}\n[End of Example {}]'.format(i+1, ex, i+1) for i,ex in enumerate(examples)]) + '-'*40
        return example_str
    
    def get_model_response(self, model_responses):
        mapping = {i: chr(ord('a') + i - 1) for i in range(1, 27)}
        if len(model_responses) == 1:
            model_response_str = model_responses[0]
        else:
            model_response_str = '\n'.join([f'[Response of Model {mapping[idx+1]}]\n{response}' for idx, response in enumerate(model_responses)])
        return model_response_str

    def generate_user_prompt(self, task, model_responses, eval_mode=None, convs=None, criteria_list=None, example_paths=None, prt=False):
        if not eval_mode:
            if len(model_responses) == 1:
                eval_mode = 'pointwise'
            elif len(model_responses) == 2:
                eval_mode = 'pairwise'
            else:
                raise ValueError('The count of model_responses = {}, which is not supported!'.format(len(model_responses)))
        template_vars = {
            "examples": self.get_examplestr(example_paths),
            "history": self.get_conv_history(convs),
            "task_description": task,
            "model_response": self.get_model_response(model_responses),
            "criteria": self.get_user_criteria(eval_mode, criteria_list),
            "evaluation_mode": eval_mode
        }
        template = Template(self._user_prompt)
        content = template.render(**template_vars)
        if prt:
            print(f'User Prompt:\n{content}')
        return content


def apply_one(prompt_path, model, task, model_responses, eval_mode, convs, prt=False):
    user_prompt = UserPrompt(prompt_path=prompt_path)
    content = user_prompt.generate_user_prompt(
        task, 
        model_responses, 
        eval_mode=eval_mode,
        convs=convs,
        criteria_list=None,
        example_paths=None,
        prt=prt
    )
    messages = [
        {'role': 'system', 'content': 'You are the Planning Agent in a multi-agent system. Your function is to design a detailed, structured evaluation plan for the given instance input. If the input contains revision feedback, revise the existing plan accordingly.'},
        {'role': 'user', 'content': content}
    ]
    clean_json_str = ''
    resp = completion(model, messages, temperature=0.7, top_p=0.8, max_try=1, prt=prt)
    # resp = completion_json(model, messages, PlannerResponse, temperature=0.8, max_try=3, prt=prt)
    if type(resp) is str:
        resp = clean_json(resp)
    try:
        parsed = PlannerResponse.model_validate_json(resp)
        clean_json_str = parsed.model_dump_json(indent=2)
    except Exception as e:
        print("Warning: PlannerResponse validation failed, return empty json str. Exception:", e)
    messages.append({'role':'assistant', 'content':clean_json_str})
    return messages, clean_json_str


def apply_batch():
    import random
    model = 'gpt-4o'
    data = load_jsonl('/Users/fuweiping/个人空间/DR/工作站/llmeval/adaptive/data/task-1000.jsonl')
    save_path = '/Users/fuweiping/个人空间/DR/工作站/llmeval/adaptive/data/plan_candidate.jsonl'
    prompt_path = 'prompts/plan.txt'

    data = data[0:]
    new_data = []
    sample_count = 3
    for idx, row in tqdm(enumerate(data), total=len(data)):
        try:
            convs = row['conversations']
            if convs and isinstance(convs, np.ndarray):
                convs = convs.tolist()
            model_responses = row['model_response']
            for i in range(sample_count):
                messages, eval_plan = apply_one(
                    prompt_path,
                    model,
                    task=row['question'],
                    model_responses=model_responses,
                    eval_mode=row['eval_type'],
                    convs=convs,
                    prt=False
                )
                # item = row.to_dict()
                row['messages'] = messages
                row['evaluation_plan'] = eval_plan
                #  print('evaluation_plan', eval_plan, type(eval_plan))
                for key, v in row.items():
                    if isinstance(v, np.ndarray):
                        row[key] = v.tolist()
                new_data.append(row)
                if len(new_data) > 0 and len(new_data) % 3 == 0:
                    save_jsonl(new_data, save_path)
        except Exception as e:
            print(f"Generate Plan Failed! Exception: {e}")
            continue
    save_jsonl(new_data, save_path)
    print(f'Save {len(new_data)} data to {save_path}!')


def append_messages(prompt_path, data_path, save_path, for_train=True):
    new_data = []
    data = load_jsonl(data_path)
    prompt = UserPrompt(prompt_path)
    for item in data:
        convs = item['conversations']
        if isinstance(convs, np.ndarray):
            convs = convs.tolist()
        content = prompt.generate_user_prompt(
            task=item['question'], 
            model_responses=item['model_response'], 
            eval_mode=item['eval_type'],
            convs=convs,
            criteria_list=None,
            example_paths=None,
            prt=True
        )
        messages = [
            {'role': 'system', 'content': 'You are the Planning Agent in a multi-agent system. Your function is to design a detailed, structured evaluation plan for the given instance input.'},
            {'role': 'user', 'content': content}
        ]
        if for_train:
            messages.append({'role': 'assistant', 'content': item['evaluation_plan']})
        new_data.append({'messages': messages})
    save_jsonl(new_data, save_path)


# ------------------------------
# Task-required aspect keywords
# ------------------------------
REQUIRED_ASPECTS = {
    "math": {
        "answer_correctness": ["answer", "final answer", "final result", "solution", "correct" "incorrect", "numerical","symbolic"],
        "reasoning_validity": ["reasoning", "logical", "valid", "sound", "derivation", "step", "mathematically", "justification"]
    },
    "coding": {
        "functional_correctness": ["functional", "correct", "output", "task requirement","specification", "solve the problem"],
        "executability": ["run", "execute", "execution", "runtime", "error", "exception","compile", "environment"]
    },
    "reasoning": {
        # "conclusion_correctness": ["conclusion", "decision", "inference", "result", "correct", "incorrect"],
        "logical_consistency": ["logical", "consistent", "reasoning", "contradiction","coherent", "sound", "flow", "step"]
    },
    "question_answering": {
        "factual_accuracy": ["factual", "factually", "accuracy", "accurate", "incorrect fact","misinformation", "contradiction", "hallucination", "contradict", "inaccurate"
        ],
        "faithfulness_grounding": ["faithful", "faithfulness", "grounded", "grounding", "source","context", "hallucination", "supported by", "evidence", "support"]
    },
    "writing": {
        # "contextual_coherence": ["coherent", "coherence", "consistent", "context", "sound","internally consistent"],
        # "relevance": ["relevant", "relevance", "addresses the prompt", "on-topic", "aligned with the prompt","responsive", "irrelevant"],
        "language_quality": ["grammar", "grammatical", "fluency", "fluent", "clarity", "language","readability", "natural", "tone"]
    },
    "dialogue": {   
        "contextual_coherence": ["coherent", "coherence", "context", "history", "consistent response"],
        "relevance": ["relevant", "relevance", "appropriate response", "addresses user intent", "on-topic"],
        "language_quality": ["grammar", "grammatical", "fluency", "fluent", "clarity", "language","readability", "natural"]
    },
    "instruction_following": {
        "instruction_compliance": ["instruction", "constraint", "requirement", "follow", "adhere","specified"],
        # "completeness": ["complete", "fully", "all parts", "missing", "partially"]
    },
    "data_analysis": {
        "data_grounding": ["data", "table", "dataset", "support", "supported by", "based on the data", "evidence"],
        "analytical_correctness": ["calculation", "analysis", "compute", "derived", "statistical", "flow"],
    },
}

# ------------------------------
# Task semantic anti-patterns
# ------------------------------
TASK_ANTI_KEYWORDS = {
    "math": ["creativity", "story", "narrative", "tone", "style"],
    "coding": ["creativity", "emotion", "story"],
    "writing": ["formal proof", "numerical accuracy", "algorithmic"],
}

def normalize_task_type(task):
    task = task.lower()
    if "math" in task:
        return "math"
    if "code" in task or "coding" in task:
        return "coding"
    if "reason" in task:
        return "reasoning"
    if "instruction" in task:
        return "instruction_following"
    if "data" in task or "table" in task:
        return "data_analysis"
    if "dialogue" in task or "chat" in task:
        return "dialogue"
    if "write" in task or "story" in task:
        return "writing"
    if "question" in task or "factual" in task:
        return "question_answering"
    return task

def collect_dimension_text(plan):
    texts = []
    for d in plan.get("evaluation_dimensions", []):
        texts.append(d.get("name", ""))
        texts.append(d.get("definition", ""))
        texts.append(d.get("rationale", ""))
        agent = d.get("assigned_agent", {})
        texts.append(agent.get("evaluation_task", ""))
        texts += agent.get("evaluation_steps")
    return " ".join(texts).lower()

def check_required_aspects(plan):
    task = normalize_task_type(plan.get("task_type", ""))
    if task not in REQUIRED_ASPECTS:
        return True, []

    text = collect_dimension_text(plan)
    missing = []
    for aspect, keywords in REQUIRED_ASPECTS[task].items():
        if not any(k in text for k in keywords):
            missing.append(aspect)

    return len(missing) == 0, missing

def semantic_sanity_check(plan):
    task = normalize_task_type(plan.get("task_type", ""))
    text = collect_dimension_text(plan) 
    for kw in TASK_ANTI_KEYWORDS.get(task, []):
        if kw in text:
            return False, kw
    return True, None

def filter_planner_data(plan):
    """
    Return:
        True  -> keep
        False -> drop
    """
    if plan is None:
        return False
    
    # 1. Required aspect coverage
    ok, missing = check_required_aspects(plan)
    if not ok:
        print(f"[DROP] missing required aspect: {missing}")
        return False

    # 2. Semantic sanity
    ok, bad_kw = semantic_sanity_check(plan)
    if not ok:
        print(f"[DROP] semantic mismatch keyword: {bad_kw}")
        return False

    return True



if __name__ == "__main__":
    # apply_batch()

    # import random
    # random.seed(42)
    # accept, revise = [], []
    # data = load_jsonl('/Users/fuweiping/个人空间/DR/工作站/llmeval/adaptive/data/plan_critic_gpt4o.jsonl')
    # for item in data:
    #     critic_result = json.loads(item['critic_result'])
    #     decision = critic_result.get('decision').lower()
    #     if decision == 'accept':
    #         evaluation_plan = json.loads(item['evaluation_plan'])
    #         if evaluation_plan.get('evaluation_mode') == item.get('eval_type'):
    #             accept.append(item)
    #             print(evaluation_plan.get('evaluation_mode'), len(item.get('model_response')))
    #     elif decision == 'revise':
    #         revise.append(item)
    #     else:
    #         print(f'Unknown decision: {decision}')
    # print(len(accept), len(revise))
  
    # unique_tasks = list(set([x['question'] for x in accept]))
    # total_size = len(unique_tasks)
    # train_size = int(total_size * 0.9)
    # train_tasks = random.sample(unique_tasks, k=train_size)
    # eval_tasks = list(set(unique_tasks) - set(train_tasks))
    # print(f"总任务数: {total_size}")
    # print(f"训练集任务数 (90%): {len(train_tasks)}")
    # print(f"验证集任务数 (10%): {len(eval_tasks)}")

    # train_data = [x for x in accept if x['question'] in train_tasks]
    # eval_data = [x for x in accept if x['question'] in eval_tasks]
    # print(f"总数据量: {len(accept)}")
    # print(f"训练集数据量: {len(train_data)}")
    # print(f"验证集数据量: {len(eval_data)}")
    
    # train_messages, eval_messages = [], []
    # for item in train_data:
    #     train_messages.append({
    #         'messages': item['messages']
    #     })
    # for item in eval_data:
    #     eval_messages.append({
    #         'messages': item['messages']
    #     })
    
    # save_jsonl(train_messages, '/Users/fuweiping/个人空间/DR/工作站/llmeval/adaptive/data/plan_sft_train.jsonl')
    # save_jsonl(eval_messages, '/Users/fuweiping/个人空间/DR/工作站/llmeval/adaptive/data/plan_sft_eval.jsonl')
    # print(train_messages[0]['messages'])
    # print(eval_messages[0]['messages'])

    # data = load_jsonl("/Users/fuweiping/个人空间/DR/工作站/llmeval/adaptive/data/for_planner.jsonl")
    # used_data = load_jsonl('/Users/fuweiping/个人空间/DR/工作站/llmeval/adaptive/data/plan_critic_gpt4o.jsonl')
    # unique_questions = list(set([x['question'] for x in data]))
    # used_questions = list(set([x['question'] for x in used_data]))
    # unique_questions = [x for x in unique_questions if x not in used_questions]
    # print(len(unique_questions))
    # import random
    # random.seed(42)
    # questions = random.sample(unique_questions, 1000)
    # new_data = []
    # for question in questions:
    #     one_item = [x for x in data if x['question']==question][0]
    #     new_data.append(one_item)
    # for item in new_data:
    #     print(item['eval_type'], len(item['model_response']), item['source'], item['category'])
    # save_jsonl(new_data, '/Users/fuweiping/个人空间/DR/工作站/llmeval/adaptive/data/for_planner-1k.jsonl')

    # import random
    # data = load_jsonl('/Users/fuweiping/个人空间/DR/工作站/llmeval/adaptive/data/for_planner.jsonl')
    # ultra = [x for x in data if x['source'] == 'ultrafeedback']
    # random.shuffle(ultra)
    # scores = []
    # for item in ultra:
    #     scores += item['score']
    # score_set = list(set(scores))
    # print(len(score_set))
    # chat = [x for x in data if x['source'] == 'chatbot-arena']
    # pairwise = ultra[:600] + chat
    # pointwise = ultra[600:]

    # pairwise = random.sample(pairwise, 300)
    # new_pairwise = []
    # for item in pairwise:
    #     label = 'tie' if item['label']=='tie (bothbad)' else item['label']
    #     new_item = {
    #         'task': item['question'],
    #         'label': label,
    #         'model_responses': item['model_response'],
    #         'eval_mode': 'pairwise',
    #         'convs': item['conversations'],
    #         'source': item['source'],
    #         'reward_weights': {
    #             'λ_c': 0.2,
    #             'alpha': 1.0,
    #             'beta': 1.0,
    #             'R': 2.0
    #         }
    #     }
    #     new_pairwise.append(new_item)


    # score_dict = {round(k, 1): 0 for k in np.arange(1.0, 10.5, 0.5)}
    # uncovered_scores = set(score_dict.keys())
    # MAX_TOTAL = 200
    # random.shuffle(pointwise)
    # priority_samples = []  # 存能覆盖新分值的项
    # residual_samples = []  # 存重复分值的项
    # # --- 第一阶段：识别贡献项 ---
    # for item in pointwise:
    #     # 注意：建议对数据里的分数也做 round(x, 1) 处理，防止浮点数精度问题导致 in 判断失败
    #     s0 = round(float(item['score'][0]), 1)
    #     s1 = round(float(item['score'][1]), 1)
        
    #     can_cover_0 = s0 in uncovered_scores
    #     can_cover_1 = s1 in uncovered_scores
        
    #     # 逻辑：只要有一个分数能覆盖新分值，就把它列为“优先”
    #     if can_cover_0 or can_cover_1:
    #         # 如果 0 能覆盖而 1 不能，选 0；反之选 1；都能则随机
    #         if can_cover_0 and not can_cover_1:
    #             idx = 0
    #         elif can_cover_1 and not can_cover_0:
    #             idx = 1
    #         else:
    #             idx = 0 if random.random() <= 0.5 else 1
                
    #         chosen_score = round(float(item['score'][idx]), 1)
    #         if chosen_score in uncovered_scores:
    #             uncovered_scores.remove(chosen_score)
            
    #         # 暂时记录到优先列表
    #         priority_samples.append((item, idx))
    #     else:
    #         # 分数已被覆盖过，存入候选池
    #         residual_samples.append(item)
    
    # # --- 第二阶段：组装并限额 ---
    # final_selected = []
    # final_selected = priority_samples[:MAX_TOTAL]
    # # 如果不满 MAX_TOTAL，从剩余项里补齐
    # if len(final_selected) < MAX_TOTAL:
    #     needed = MAX_TOTAL - len(final_selected)
    #     additional_items = random.sample(residual_samples, min(needed, len(residual_samples)))
    #     for item in additional_items:
    #         idx = 0 if random.random() <= 0.5 else 1
    #         final_selected.append((item, idx))

    # new_pointwise = []
    # for item, idx in final_selected:
    #     chosen_score = float(item['score'][idx])
    #     r_score = round(chosen_score, 1)
    #     if r_score in score_dict:
    #         score_dict[r_score] += 1
    #     new_item = {
    #         'task': item['question'],
    #         'label': chosen_score,
    #         'model_responses': [item['model_response'][idx]],
    #         'eval_mode': 'pointwise',
    #         'convs': item['conversations'],
    #         'source': item['source'],
    #         'reward_weights': {
    #             'λ_c': 0.2,
    #             'alpha': 1.0,
    #             'beta': 1.0,
    #             'R': 2.0
    #         }
    #     }
    #     new_pointwise.append(new_item)
    
    # print(len(new_pairwise), len(new_pointwise))
    # print(set([x['label'] for x in new_pairwise]), set([len(x['model_responses']) for x in new_pairwise]), set([x['eval_mode'] for x in new_pairwise]))
    # print(set([x['label'] for x in new_pointwise]), len(set([x['label'] for x in new_pointwise])), set([len(x['model_responses']) for x in new_pointwise]), set([x['eval_mode'] for x in new_pointwise]), score_dict)

    
    # samples = new_pairwise + new_pointwise
    # random.shuffle(samples)
    # print(len(set([x['task'] for x in samples])))
    # save_jsonl(samples, '/Users/fuweiping/个人空间/DR/工作站/llmeval/adaptive/data/grpo_r1.jsonl')

    # data = load_jsonl('/Users/fuweiping/个人空间/DR/工作站/llmeval/adaptive/data/grpo_r1.jsonl')
    # result = {k: [item[k] for item in data] for k in data[0]}
    # for k, v in result.items():
    #     print(k, len(v), v[0])
    # label_list = [str(x) for x in result['label']]
    # result['label'] = label_list

    # import json
    # with open('/Users/fuweiping/个人空间/DR/工作站/llmeval/adaptive/data/grpo_r1.json', 'w', encoding='utf-8') as f:
    #     json.dump(result, f, ensure_ascii=False, indent=4)
    # print(result['label'])

    prompt_path = './prompts/plan.txt'
    data_path = '/Users/fuweiping/个人空间/DR/工作站/llmeval/adaptive/data/grpo_r1.jsonl'
    new_data = []
    data = load_jsonl(data_path)
    prompt = UserPrompt(prompt_path)
    for item in data:
        convs = item['convs']
        if isinstance(convs, np.ndarray):
            convs = convs.tolist()
        content = prompt.generate_user_prompt(
            task=item['task'], 
            model_responses=item['model_responses'], 
            eval_mode=item['eval_mode'],
            convs=convs,
            criteria_list=None,
            example_paths=None,
            prt=True
        )
        messages = [
            {'role': 'system', 'content': 'You are the Planning Agent in a multi-agent system. Your function is to design a detailed, structured evaluation plan for the given instance input.'},
            {'role': 'user', 'content': content}
        ]
        item['messages'] = messages
        item['label'] = str(item['label'])
        new_data.append(item)
    save_jsonl(new_data, '/Users/fuweiping/个人空间/DR/工作站/llmeval/adaptive/data/grpo_r1-new.jsonl')