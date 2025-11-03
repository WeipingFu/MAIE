import os
import json
from typing import List, Dict, Tuple
from plan import PlanAgent
from critic import CriticAgent
from judge import JudgeAgent
from check_result import validate_plan_json
from parse_result import parse_evaluation_plan
from utils import load_json, save_json, save_jsonl 
import copy
from tqdm import tqdm
import pandas as pd


# Generate Plan
def generate_plan(plan_agent: PlanAgent, task: str, model_responses: List[str], convs: list = None, round: int = 1, feedback: dict = None, pre_messages: list = None, max_try: int = 3) -> Tuple[dict, str]:
    """
    Generate and validate the plan structure.
    Returns:
        plan_json, status_msg, structure_fail_count
    """
    structure_fail_count = 0
    for attempt in range(1, max_try + 1):
        print(f"\n[PLAN] Attempt {attempt}/{max_try} generating plan...")
        plan_resp = plan_agent.apply_one(task, model_responses, convs, round, feedback, pre_messages)
        try:
            plan_json = json.loads(plan_resp)
        except json.JSONDecodeError:
            print("Plan JSON parsing failed. Retrying...")
            structure_fail_count += 1
            continue

        # validate structure
        if round == 1:
            is_valid, msg = validate_plan_json(plan_json)
        else:
            is_valid, msg = validate_plan_json(plan_json.get('revised_plan'))

        if is_valid:
            print(f"Plan structure validated successfully on attempt {attempt}.")
            return plan_json, "success"
        else:
            print(f"Plan validation failed: {msg}")
            structure_fail_count += 1

    return None, "Plan structure fail after max_try.", structure_fail_count


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
        print("Critic feedback:", critic_resp)
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
def parse_plan_and_config_judges(plan_json: dict, save_dir: str) -> List[str]:
    """
    Parse the plan and generate configuration files for each judge.
    Return a list of configuration paths for each judge.
    """
    os.makedirs(save_dir, exist_ok=True)
    parse_evaluation_plan(plan_json, save_dir=save_dir, strategy='heuristic')
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
    eval_mode = plan_json.get("evaluation_mode", "pointwise").lower()
    detailed_scores = {}

    if eval_mode == "pointwise":
        total_score = 0.0
        weight_sum = 0.0
        for dim in dims:
            name = dim["name"]
            weight = float(dim.get("weight", 0))
            result_json = judge_results["round2"].get(name, "{}")
            try:
                result = json.loads(result_json)
                score = float(result.get("judgement", 0))
            except Exception:
                score = 0.0
            total_score += weight * score
            weight_sum += weight
            detailed_scores[name] = {"weight": weight, "score": score}

        final_score = total_score / weight_sum if weight_sum > 0 else 0.0
        # return {
        #     "evaluation_mode": "pointwise",
        #     "final_score": final_score,
        #     "judgement": "",  # no winner concept in pointwise
        #     "details": detailed_scores
        # }
        return final_score

    elif eval_mode == "pairwise":
        response_scores = {"response 1": 0, "response 2": 0}
        weight_sum = 0.0

        for dim in dims:
            name = dim["name"]
            weight = float(dim.get("weight", 0))
            result_json = judge_results["round2"].get(name, "{}")
            try:
                result = json.loads(result_json)
                judgement = result.get("judgement", "").strip().lower()
            except Exception:
                judgement = ""

            if judgement.lower() == "tie":
                response_scores["response 1"] += weight / 2
                response_scores["response 2"] += weight / 2
            else:
                response_scores[judgement] += weight

            weight_sum += weight
            detailed_scores[name] = {"weight": weight, "judgement": judgement}

        if weight_sum > 0:
            response_scores = {k: v / weight_sum for k, v in response_scores.items()}
        if abs(response_scores["response 1"] - response_scores["response 2"]) < 1e-6:
            final_judgement = "Tie"
        elif response_scores["response 1"] > response_scores["response 2"]:
            final_judgement = "Response 1"
        else:
            final_judgement = "Response 2"
        # return {
        #     "evaluation_mode": "pairwise",
        #     "final_score": response_scores,
        #     "judgement": final_judgement,
        #     "details": detailed_scores
        # }
        return final_judgement
    else:
        raise ValueError(f"Unknown evaluation mode: {eval_mode}")


# pipeline
def run_full_evaluation(task: str, model_responses: List[str], convs: list = None, plan_agent_args: str = './args/plan.json', revise_agent_args: str = './args/plan-revise.json', critic_args: str = './args/plan-critic.json', save_dir: str = "./args/judge_configs", max_revise_round: int = 3, judge_rounds: int = 2, max_try: int = 3):
    plan_agent = PlanAgent(plan_agent_args)
    critic_agent = CriticAgent(critic_args)

    stats = {
        "plan_structure_failures": 0,
        "critic_first_accept": False,
        "revise_rounds": 0
    }

    # Step 1: Generate initial plan
    origin_plan_json, msg, structure_fails = generate_plan(plan_agent, task, model_responses, convs=convs, max_try=max_try)
    stats["plan_structure_failures"] += structure_fails
    if not origin_plan_json:
         return {"judgement": None, "judge_plan": None, "judge_details": None, "stats": stats, "reason": msg}

    # Step 2: Review plan and revise
    pre_messages = plan_agent.messages + [{'role': 'assistant', 'content': json.dumps(origin_plan_json, indent=4, ensure_ascii=False)}]
    plan_json = copy.deepcopy(origin_plan_json)
    for i in range(1, max_revise_round+1):
        plan_json, critic_result, msg = review_plan(critic_agent, task, model_responses, plan_json, max_try=max_try)
        # Critic fail or accept, break and use original plan
        if not critic_result or msg == 'accept':
            if i == 1 and msg == 'accept':
                stats["critic_first_accept"] = True
            break
        elif msg == 'revise':
            stats["revise_rounds"] += 1
            revise_agent = PlanAgent(revise_agent_args)
            revise_result, msg, structure_fails = generate_plan(
                revise_agent, task, model_responses, convs,
                round=i+1, feedback=critic_result, 
                pre_messages=pre_messages, max_try=max_try
            )
            stats["plan_structure_failures"] += structure_fails
            # Revise fail, break and use original plan
            if not revise_result:
                break
            plan_json = revise_result.get('revised_plan', revise_result)
            pre_messages = revise_agent.messages + [{'role': 'assistant', 'content': json.dumps(revise_result, indent=4, ensure_ascii=False)}]

    # Step 3: Parse plan
    judge_files = parse_plan_and_config_judges(plan_json, save_dir)

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
def run_evaluation_batch(data: list, result_save_path: str, plan_agent_args: str = './args/plan.json', revise_agent_args: str = './args/plan-revise.json', critic_args: str = './args/plan-critic.json', config_save_dir: str = './args/judge_configs', max_revise_round: int = 3, judge_rounds: int = 2, max_try: int = 3):
    new_data = []
    for idx, one in tqdm(enumerate(data), total=len(data)):
        output = run_full_evaluation(
            task=one.get('task'),
            model_responses=one.get('model_responses'),
            plan_agent_args=plan_agent_args,
            revise_agent_args=revise_agent_args,
            critic_args=critic_args,
            save_dir=config_save_dir+'/'+str(idx),
            max_revise_round=max_revise_round,
            judge_rounds=judge_rounds,
            max_try=max_try
        )
        for key, v in output.items():
            one[key] = v
        new_data.append(one)
        if result_save_path and len(new_data) % 10 == 0:
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
    #     plan_agent_args="./args/plan.json",
    #     revise_agent_args="./args/plan-revise.json",
    #     critic_args="./args/plan-critic.json",
    #     save_dir="./judge_configs",
    #     max_revise_round=3,
    #     max_try=3
    # )
    # print("\n=== FINAL EVALUATION RESULT ===")
    # print(json.dumps(output, indent=2, ensure_ascii=False))

    data = ''
    result_save_path = ''
    run_evaluation_batch(
        data=data, 
        result_save_path=result_save_path, 
        plan_agent_args='./args/plan.json', 
        revise_agent_args='./args/plan-revise.json', 
        critic_args='./args/plan-critic.json', 
        config_save_dir='./args/mt-bench/judge_configs', 
        max_revise_round=3, 
        judge_rounds=1, 
        max_try=3
    )
