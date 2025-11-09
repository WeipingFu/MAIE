from utils import load_json, save_json
import random
random.seed(42)

# Tier 3: Flagship models, excel at complex reasoning, multi-step mathematics, and understanding nuanced context.
# Tier 2: Elite models, excel at most common tasks and logical reasoning.
# Tier 1: Basic models, excel at simple fact-checking and formatting.
# JUDGE_MODEL_METADATA = {
#     "gpt-4o": {"tier": 3, "cost": 10.0, "task_type": "common", "model_family": "gpt", "model_type": "close", "model_path":"", "tokenizer_path":""},
# }

JUDGE_POOL = load_json('./args/judge_pool.json')

def get_model_score_heuristic(dimension: dict, model_name: str) -> float:
    """
    - Calculates the matching score between the model and the evaluation dimensions.
    - The matching score takes into account: dimension weight (importance), dimension complexity, and the model's tier and cost.
    - Goal: Models with strong capabilities (higher tiers) will score higher, while also considering cost appropriately (lower cost is better).
    """

    model_meta = JUDGE_POOL.get(model_name, {"tier": 1, "cost": 1.0})
    
    # dimension importance and complexity
    weight = float(dimension.get("weight", 0.3))
    complex_keywords = ["reasoning", "logic", "consistency", "factual", "depth", "creativity", "math", "code"]
    # check if the dimension name or definition contain complex keywords
    dim_text = dimension.get("name", "") + " " + dimension.get("definition", "")
    complexity_boost = 0.0
    if any(k in dim_text for k in complex_keywords):
        complexity_boost = 0.2
    
    dimension_score = weight * (1 + complexity_boost)
    
    # Tier / Cost
    # Use Tier / sqrt(Cost) to balance capacity and cost.
    model_match_score = model_meta["tier"] / (model_meta["cost"] ** 0.5)

    # final score
    final_score = dimension_score * model_match_score
    
    return final_score


def parse_evaluation_plan(plan_json: dict, save_dir:str, system_prompt_path: str = './prompts/judge-r1-system.txt', user_prompt_path: str = './prompts/judge-r1-user.txt', model_type: str = 'open', strategy: str = "random") -> None:
    """
    Parses the evaluation plan and assigns a Judge model to each dimension 
    based on the specified strategy ('random' or 'heuristic') and model filtering.

    Args:
        plan_json: The complete evaluation plan JSON dictionary.
        model_type: Filter for 'open' (open-source) or 'close' (closed-source) models. Empty for both.
        strategy: 'random' for random assignment with reuse, 'heuristic' for best-fit assignment.
   
    plan_json
    {
        "task_type": "",
        "sub_task": "",
        "evaluation_mode": "",
        "evaluation_dimensions": [
            {"name":"", "definition":"", "rationale":"", "weight":"0.2", "evaluation_granularity": "",
            "scoring_scale":"", "high_score_indicator":"", "low_score_indicator":"", "dependencies":[],
            "assigned_agent":{
                "role_name":"", "role_description":"", "evaluation_task":"", "evaluation_steps":[]
            }}, ...
        ],
        "evaluation_graph": {"nodes":[], "edges":[{"from":"", "to":""}, ...]}
    }
    """
   
    task_type = plan_json.get("task_type", "common").lower()
    eval_dims = plan_json.get("evaluation_dimensions", [])

    # 1. Filter available judges by task_type (allows common models for specific tasks)
    available_judges_meta = {
        model: attr for model, attr in JUDGE_POOL.items() 
        if attr.get("task_type") == task_type or attr.get("task_type") == "common"
    }

    # 2. Filter by model_type ('open' or 'close')
    if model_type:
        model_type_lower = model_type.lower()
        available_judges_meta = {
            model: attr for model, attr in available_judges_meta.items() 
            if attr.get("model_type") == model_type_lower
        }
    available_models = list(available_judges_meta.keys())
    if not available_models:      # Fallback to a safe default if no judges are found
        print(f"Warning: No judges available for task type '{task_type}' and model type '{model_type}'. Using default.")
        available_models = ["llama-3.1-8b-instruct"] 
        available_judges_meta = {"llama-3.1-8b-instruct": JUDGE_POOL.get("llama-3.1-8b-instruct", {"model_type": "open"})}

    # 3. Assign models to dimensions
    final_assignments = {}      # {dimension_index: model_name}
    if strategy == "random":    
        # random assignment
        model_assignments = random.choices(available_models, k=len(eval_dims))
        for dim_index, _ in enumerate(eval_dims):
            final_assignments[dim_index] = model_assignments[dim_index]
    elif strategy == "heuristic": 
        # Heuristic assignment   
        for dim_index, dimension in enumerate(eval_dims):
            best_score = -1
            best_model_name = None
            # Find the best model for this specific dimension
            for model_name in available_models:
                score = get_model_score_heuristic(dimension, model_name)
                if score > best_score:
                    best_score = score
                    best_model_name = model_name
            if best_model_name:
                final_assignments[dim_index] = best_model_name
    else:
        # Fallback for unknown strategy
        print(f"Warning: Unknown strategy '{strategy}'. Defaulting to 'random'.")
        return parse_evaluation_plan(plan_json, model_type, "random")
    
    # 4. Build Arg Files based on final assignments
    for dim_index, model_name in final_assignments.items():
        dimension = eval_dims[dim_index]
        model_meta = available_judges_meta.get(model_name)
        agent_file = {
            "system_prompt_path": system_prompt_path,
            "user_prompt_path": user_prompt_path,
            "task_type": task_type,
            "evaluation_mode": plan_json.get("evaluation_mode", ""),
            "dimension": dimension['name'],
            'definition': dimension['definition'],
            "scoring_scale": dimension['scoring_scale'],
            "high_score_indicator": dimension['high_score_indicator'],
            "low_score_indicator": dimension['low_score_indicator'],
            "model_name": model_name,
            "model_type": model_meta.get("model_type", ""),
            "model_path": model_meta.get("model_path", ""),
            "tokenizer_path": model_meta.get("tokenizer_path", ""),
            "role_name": dimension["assigned_agent"].get("role_name", f"{dimension['name']} Judge"),
            "role": dimension["assigned_agent"].get("role_description", f"You are a specialized agent to evaluate the '{dimension['name']}' dimension."),
            "task": dimension["assigned_agent"].get("evaluation_task", f"Evaluate the generated content based on the definition: {dimension['definition']}"),
            "granularity": dimension['evaluation_granularity'],
            "steps": dimension["assigned_agent"].get("evaluation_steps", "1. Read the definition and scoring scale. 2. Compare the content with high/low score indicators. 3. Provide score, rationale, and evidence."),
            "dependencies": dimension['dependencies']
        }
        save_path = save_dir+f"{dimension['name']}_judge.json"
        save_json(agent_file, save_path)
        print('Save judge\'s arg file to {}'.format(save_path))
    save_path = save_dir+'plan.json'
    save_json(plan_json, save_path)
    print(f'Save plan to {save_path}')


if __name__ == "__main__":
    plan_json = load_json('../../result/testcase/math-judgebench/plan-0shot-llama3.1.json')
    save_dir = '../../result/testcase/math-judgebench/'
    parse_evaluation_plan(plan_json, save_dir, model_type='open', strategy='heuristic')