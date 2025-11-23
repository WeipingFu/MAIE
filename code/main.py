from agents.PlanAgent import PlannerAgent
from agents.CriticAgent import CriticAgent
from agents.JudgeAgent import JudgeAgent
from autogen_agentchat.messages import UserMessage
import json

# aggregate judge results
def aggregate_final_result(plan_json, judge_results):
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
            result_json = judge_results.get(name, "{}")
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
            weight = float(dim.get("weight", 0))
            result_json = judge_results.get(name, "{}")
            result_json = result_json.replace('```json','').replace('```','')
            # print("result json*********", result_json)
            try:
                result = json.loads(result_json)
                judgement = result.get("judgement", "").strip().lower()
            except Exception:
                judgement = ""
            print(f'Dimension: {name}, Judgement: {judgement}')
            if judgement.lower() == "tie":
                response_scores["response 1"] += weight / 2
                response_scores["response 2"] += weight / 2
            else:
                response_scores[judgement] += weight

            weight_sum += weight
            detailed_scores[name] = {"weight": weight, "judgement": judgement}
        
        if weight_sum > 0:
            response_scores = {k: v / weight_sum for k, v in response_scores.items()}
        # print(f'response_scores:{response_scores} =====================================================')
        if abs(response_scores["response 1"] - response_scores["response 2"]) < 1e-6:
            final_judgement = "Tie"
        elif response_scores["response 1"] > response_scores["response 2"]:
            final_judgement = "Response 1"
        else:
            final_judgement = "Response 2"
        print(f'Final Judgement: {final_judgement}')
        return final_judgement
    else:
        raise ValueError(f"Unknown evaluation mode: {eval_mode}")


async def run_pipeline(task, model_responses, convs=None, with_critic=True, judge_chat=True):
    print("--------------------Start Evaluation--------------------")
    print(f"Enable Critic: {with_critic}\nEable Multi-round Judge: {judge_chat}")

    # Initialize PlannerAgent
    planner = PlannerAgent(name='planner', mode='plan')

    planner_response = await planner.on_messages(
        UserMessage(content={
            "task": task,
            "model_responses": model_responses,
            "convs": convs
        }), 
    )

    # If enable critic, then start planner-critic group chat
    # If not enable critic, then parse planner response and assign judges
    if with_critic:
        pass
    
    evaluation_plan = planner_response.chat_message.content
    dimensions = evaluation_plan.evaluation_dimensions
    dimension_names = [d['name'] for d in dimensions]
    print(f"--------------------Evaluate {len(dimensions)} Dimensions--------------------")
    print(f"Dimensions are: {dimension_names}")

    # Initialize JudgeAgent
    judge = JudgeAgent(name='judge')
    judge_results = {k['name']:None for k in dimensions}
    # Judge each dimension independently
    for dimension_plan in dimensions:
        dimension_name = dimension_plan['name']
        judge_response = await judge.on_messages(
            UserMessage(content={
                "task": task,
                "model_responses": model_responses,
                "dimension_plan": dimension_plan,
                "convs": convs,
                "example_paths": None
            })
        )
        judge_results[dimension_name] = judge_response.chat_message.content
    # If enable judge chat, then start judges' group chat
    # If not enable judge chat, then aggreate judge results to final result
    final_judgement = aggregate_final_result(evaluation_plan, judge_results)
    print('--------------------End of Evaluation--------------------')
    return judge_results, final_judgement

if __name__ == "__main__":
    from tqdm import tqdm
    import pandas as pd
    mtbench = pd.read_excel('')
    for idx, row in tqdm(mtbench.iterrows()):
        task = row['question']

