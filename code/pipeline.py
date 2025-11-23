import os
import json
from typing import List, Dict, Tuple
from plan import PlanAgent
from critic import CriticAgent
from judge import JudgeAgent
from check_result import validate_plan_json
from parse_result import parse_evaluation_plan
from utils import load_json, save_json, load_jsonl, save_jsonl 
import copy
from tqdm import tqdm
import pandas as pd
# from model_scheduler import ModelScheduler, GLOBAL_SCHEDULER
from copy import deepcopy

# Generate Plan
def generate_plan(plan_agent: PlanAgent, task: str, model_responses: List[str], convs: list = None, round: int = 1, feedback: dict = None, pre_messages: list = None, attempt: int = 1) -> Tuple[dict, str]:
    """
    Generate and validate the plan structure.
    Returns:
        plan_json, errors, status_msg
    """
    print(f"\n[PLAN] Attempt {attempt} generating plan...")
    plan_resp = plan_agent.apply_one(task, model_responses, convs, round, feedback, pre_messages)
    errors = None
    plan_resp = plan_resp.replace('```json','').replace('```','')
    try:
        plan_json = json.loads(plan_resp)
    except json.JSONDecodeError:
        print(f"Plan JSON parsing failed. Retrying...")
        print(plan_resp)
        return None, errors, 'not a json'
        
    # validate structure
    if round == 1:
        is_valid, errors, msg = validate_plan_json(plan_json)
    else:
        is_valid, errors, msg = validate_plan_json(plan_json.get('revised_plan'))

    if is_valid:
        print(f"Plan structure validated successfully at Attempt {attempt}.")
        return plan_json, errors, "success"
    else:
        print(f"Plan validation failed: {msg}. Try again...")
        return plan_json, errors, 'fail'


# Plan Critic
def review_plan(critic_agent: CriticAgent, task: str, model_responses: List[str], plan_json: dict, max_try: int = 3) -> Tuple[dict, str]:
    """
    The plan is reviewed and revised with the help of a critic.
    Returns:
        (plan_json, critic_result, decision_msg)
    """
    for attempt in range(1, max_try + 1):
        print(f"\n[CRITIC] Attempt {attempt}/{max_try} reviewing plan...")
        critic_resp = critic_agent.apply_one(task, model_responses, plan_json)
        critic_resp = critic_resp.replace('```json','').replace('```','')
        # print("Critic feedback:", critic_resp)
        # structure error
        try:
            critic_result = json.loads(critic_resp)
        except json.JSONDecodeError:
            print("Critic output is not valid JSON, continue critic.")
            critic_result = None
            continue
        decision = critic_result.get("decision", "").lower()
        # accept
        if decision in ["accept", "pass", "approve"]:
            print(f"Plan approved by critic on attempt {attempt}.")
            return plan_json, critic_result, "accept"
        # revise
        elif decision in ["revise", "reject"]:
            print("Critic rejected the plan. Revising...")
            return plan_json, critic_result, "revise"

    return plan_json, None, "Plan Critic fail after max_try."


# Parse Plan
def parse_plan_and_config_judges(plan_json: dict, save_dir: str, model_type: str = 'open') -> List[str]:
    """
    Parse the plan and generate configuration files for each judge.
    Return a list of configuration paths for each judge.
    """
    os.makedirs(save_dir, exist_ok=True)
    parse_evaluation_plan(plan_json, save_dir=save_dir, model_type=model_type, strategy='heuristic')
    judge_files = [os.path.join(save_dir, f) for f in os.listdir(save_dir) if f.endswith("_judge.json")]
    print(f"Generated {len(judge_files)} judge configs.")
    return judge_files


# Run Judge Agents (two turns)
def run_judge_rounds(task: str, model_responses: List[str], judge_files: List[str], convs: list = None, rounds: int = 2) -> Dict[str, dict]:
    """
    Perform two rounds of evaluation for each judge agent.
        - First round: Independent evaluation
        - Second round: Integration based on dependencies
    """
    results_round1 = {}
    results_round2 = {}

    # Turn 1
    for file_path in judge_files:
        judge_args = load_json(file_path)
        dim_name = judge_args["dimension"]
        print(f"\n[JUDGE-R1] Evaluating dimension: {dim_name}")
        judge_agent = JudgeAgent(file_path)
        resp = judge_agent.apply_one(task, model_responses, convs, round=1)
        results_round1[dim_name] = resp

    if rounds < 2:
        return {"round1": results_round1, "round2": results_round1}

    # Turn 2
    for file_path in judge_files:
        judge_args = load_json(file_path)
        dim_name = judge_args["dimension"]
        depends = judge_args.get("dependencies", [])
        print(f"\n[JUDGE-R2] Re-evaluating dimension: {dim_name} with dependencies: {depends}")
        judge_agent = JudgeAgent(file_path)
        dep_feedback = {dep: results_round1.get(dep, "") for dep in depends}
        # judge_agent.params["dependencies_feedback"] = dep_feedback
        # resp = judge_agent.apply_one(task, model_responses, round=2)
        results_round2[dim_name] = resp

    return {"round1": results_round1, "round2": results_round2}


# aggregate results
def aggregate_final_result(plan_json: dict, judge_results: Dict[str, dict]) -> dict:
    """
    Aggregate final results based on evaluation mode.
    - If mode == "pointwise": weighted average of scores.
    - If mode == "pairwise": compute response-wise win rates.
    """
    dims = plan_json.get("evaluation_dimensions", [])
    eval_mode = plan_json.get("evaluation_mode", "").lower()
    detailed_scores = {}

    if eval_mode == "pointwise":
        total_score = 0.0
        weight_sum = 0.0
        for dim in dims:
            name = dim["name"]
            weight = float(dim.get("weight", 0))
            result_json = judge_results["round2"].get(name, "{}")
            result_json = result_json.replace('```json','').replace('```','')
            try:
                result = json.loads(result_json)
                score = float(result.get("judgement", 0))
            except Exception:
                score = 0.0
            total_score += weight * score
            weight_sum += weight
            detailed_scores[name] = {"weight": weight, "score": score}

        final_score = total_score / weight_sum if weight_sum > 0 else 0.0
        return final_score

    elif eval_mode == "pairwise":
        response_scores = {"response 1": 0, "response 2": 0}
        weight_sum = 0.0

        for dim in dims:
            name = dim["name"]
            print(f'Dimension Name: {name} =======================================================')
            weight = float(dim.get("weight", 0))
            result_json = judge_results["round2"].get(name, "{}")
            result_json = result_json.replace('```json','').replace('```','')
            print("result json*********", result_json)
            try:
                result = json.loads(result_json)
                print("result*******", result)
                judgement = result.get("judgement", "").strip().lower()
            except Exception:
                judgement = ""
            print(f'Judgement:{judgement} =====================================================')
            if judgement.lower() == "tie":
                response_scores["response 1"] += weight / 2
                response_scores["response 2"] += weight / 2
            else:
                response_scores[judgement] += weight

            weight_sum += weight
            detailed_scores[name] = {"weight": weight, "judgement": judgement}
        
        if weight_sum > 0:
            response_scores = {k: v / weight_sum for k, v in response_scores.items()}
        print(f'response_scores:{response_scores} =====================================================')
        if abs(response_scores["response 1"] - response_scores["response 2"]) < 1e-6:
            final_judgement = "Tie"
        elif response_scores["response 1"] > response_scores["response 2"]:
            final_judgement = "Response 1"
        else:
            final_judgement = "Response 2"
        print(f'final_judgement:{final_judgement} =====================================================')
        return final_judgement
    else:
        raise ValueError(f"Unknown evaluation mode: {eval_mode}")


# pipeline
def run_full_evaluation(task: str, model_responses: List[str], convs: list = None, plan_agent_args: str = './args/plan.json', revise_agent_args: str = './args/plan-revise.json', critic_args: str = './args/plan-critic.json', save_dir: str = "./args/judge_configs", max_revise_round: int = 3, judge_rounds: int = 2, judge_type: str = 'open', max_try: int = 3):
    
    critic_agent = CriticAgent(critic_args)

    stats = {
        "plan_structure_failures": 0,
        "critic_first_accept": False,
        "revise_rounds": 0
    }

    # Step 1: Generate initial plan and validate structure
    plan_agent = PlanAgent(plan_agent_args)
    origin_plan_json, errors, msg = generate_plan(plan_agent, task, model_responses, convs=convs, round=1, attempt=1)
    feedback = None
    pre_messages = []
    for attempt in range(2, max_try+1):
        if msg == 'success':
            break
        elif msg == 'not a json':
            stats["plan_structure_failures"] += 1
            origin_plan_json, errors, msg = generate_plan(plan_agent, task, model_responses, convs=convs, round=1, attempt=attempt)
        elif msg == 'fail':
            pre_messages = plan_agent.messages + [{'role': 'assistant', 'content': json.dumps(origin_plan_json, indent=4, ensure_ascii=False)}]
            feedback = {
                "decision": "revise",
                "confidence": "1",
                "issues": errors,
                "suggestion": "Modify the issues find in Structure.",
                "rationale": "The output JSON structure is invalid according to the required schema."
            }
            structure_revise_agent = PlanAgent(revise_agent_args)
            origin_plan_json, errors, msg = generate_plan(structure_revise_agent, task, model_responses, convs=convs, round=2, feedback=feedback, pre_messages=pre_messages, attempt=attempt)
    # fail after may_try     
    if msg != 'success':
        return {"judgement": None, "judge_plan": None, "judge_details": None, "stats": stats, "reason": 'Initial plan generate fail.'}

    # Step 2: Review plan and revise
    pre_messages = plan_agent.messages + [{'role': 'assistant', 'content': json.dumps(origin_plan_json, indent=4, ensure_ascii=False)}]
    plan_json = copy.deepcopy(origin_plan_json)
    critic_result = None
    for i in range(1, max_revise_round+1):
        if plan_json:
            plan_json, critic_result, msg = review_plan(critic_agent, task, model_responses, plan_json, max_try=max_try)
        # Critic fail or accept, break and use original plan
        if not critic_result or msg == 'accept':
            if i == 1 and msg == 'accept':
                stats["critic_first_accept"] = True
            break
        elif msg == 'revise':
            stats["revise_rounds"] += 1
            revise_agent = PlanAgent(revise_agent_args)
            revise_result, errors, msg = generate_plan(
                revise_agent, task, model_responses, convs,
                round=i+1, feedback=critic_result, 
                pre_messages=pre_messages
            )
            # Revise fail, continue generating plan
            if msg != 'success':
                stats["plan_structure_failures"] += 1
                continue
            # Revise success, continue reviewing plan
            plan_json = revise_result.get('revised_plan', revise_result)
            pre_messages = revise_agent.messages + [{'role': 'assistant', 'content': json.dumps(revise_result, indent=4, ensure_ascii=False)}]

    # Step 3: Parse plan
    judge_files = parse_plan_and_config_judges(plan_json, save_dir, model_type=judge_type)

    # Step 4: Run Judging
    judge_results = run_judge_rounds(task, model_responses, judge_files, rounds=judge_rounds)

    # Step 5: Aggregate judgements
    final_result = aggregate_final_result(plan_json, judge_results)

    return {
        "judgement": final_result,
        "judge_plan": plan_json,
        "judge_details": judge_results,
        "stats": stats,
        "reason": "success"
    }


# save result
def save_results(data: list, save_path: str):
    if '.jsonl' in save_path:
        save_jsonl(data, save_path)
    elif '.json' in save_path:
        save_json(data, save_path)
    elif '.xlsx' in save_path:
        pd.DataFrame(data).to_excel(save_path, index=False)
    else:
        print('Save results fail due to unsupported format!')
        return
    print(f'Save {len(data)} results to: {save_path}')


# batch apply
def run_evaluation_batch(data: list, result_save_path: str, plan_agent_args: str = './args/plan.json', revise_agent_args: str = './args/plan-revise.json', critic_args: str = './args/plan-critic.json', config_save_dir: str = './args/judge_configs', max_revise_round: int = 3, judge_rounds: int = 2, judge_type: str = 'open', max_try: int = 3):
    # global GLOBAL_SCHEDULER
    # if GLOBAL_SCHEDULER is None:
    #     GLOBAL_SCHEDULER = ModelScheduler() 
    #     print(f"GLOBAL_SCHEDULER initialized successfully. Target Device: {GLOBAL_SCHEDULER.device}")

    new_data = []
    for idx, one in tqdm(enumerate(data), total=len(data)):
        output = {"judgement": None, "judge_plan": None, "judge_details": None, "stats": None, "reason": 'error'}
        try:
            output = run_full_evaluation(
                task=one.get('task'),
                model_responses=one.get('model_responses'),
                convs=one.get('convs'),
                plan_agent_args=plan_agent_args,
                revise_agent_args=revise_agent_args,
                critic_args=critic_args,
                save_dir=config_save_dir+'/'+str(idx)+'/',
                max_revise_round=max_revise_round,
                judge_rounds=judge_rounds,
                judge_type=judge_type,
                max_try=max_try
            )
        except Exception as e:
            output['reason'] = e
        for key, v in output.items():
            one[key] = v
        new_data.append(one)
        if result_save_path and len(new_data) % 1 == 0:
            save_results(new_data, result_save_path)

    if result_save_path:
        save_results(new_data, result_save_path)
        


if __name__ == "__main__":
    # task = "Evaluate the quality and correctness of model responses for the math reasoning task."
    # model_responses = [
    #     "The model reasoned step-by-step and got the correct answer 42.",
    #     "The model skipped key reasoning and output 40 without explanation."
    # ]
    # output = run_full_evaluation(
    #     task=task,
    #     model_responses=model_responses,
    #     convs=None,
    #     plan_agent_args="./args/plan.json",
    #     revise_agent_args="./args/plan-revise.json",
    #     critic_args="./args/plan-critic.json",
    #     save_dir="./args/judge_configs",
    #     max_revise_round=3,
    #     judge_rounds=1, 
    #     max_try=3
    # )
    # print("\n=== FINAL EVALUATION RESULT ===")
    # print(json.dumps(output, indent=2, ensure_ascii=False))


    # test_data = load_jsonl('/data/fwp/workspace/benchmarks/mt-bench-human/human.jsonl')
    test_data = pd.read_excel('/data/fwp/workspace/adaptive/result/mt-bench/sample-vanilla-llama3.1.xlsx').to_dict(orient='records')
    print(len(test_data))
    data = []
    for idx, row in enumerate(test_data):
        one = {'task': row['question'], 'model_responses':[row['model_a_response'], row['model_b_response']], 'convs':None}
        if int(row['turn']) > 1:
            
            conv_a, conv_b = eval(row['conversation_a'])[:-2], eval(row['conversation_b'])[:-2]
            convs = []
            for i in range(0, len(conv_a), 2):
                conv = {
                    'user': conv_a[i]['content'],
                    'a': conv_a[i+1]['content'],
                    'b': conv_b[i+1]['content']
                }
                convs.append(conv)
            one['convs'] = convs
        data.append(one)
    # print(data[0])
    result_save_path = '../../result/mt-bench/sample-gpt4o.xlsx'
    print("\nStarting batch evaluation...")
    run_evaluation_batch(
        data=data[0:], 
        result_save_path=result_save_path, 
        plan_agent_args='./args/plan.json', 
        revise_agent_args='./args/plan-revise.json', 
        critic_args='./args/plan-critic.json', 
        config_save_dir='./args/mt-bench/judge_gpt4o', 
        max_revise_round=2, 
        judge_rounds=1,
        judge_type='close',
        max_try=3
    )
