from .agents.PlanAgent import PlannerAgent
# from .agents.CriticAgent import CriticAgent
from .agents.JudgeAgent import JudgeAgent
from autogen_agentchat.messages import StructuredMessage
import json
from typing import List, Optional
from pydantic import BaseModel
from autogen_core import CancellationToken

def safe_load_json(content):
    if type(content) is str:
        content = content.strip().replace('```json','').replace('```','')
        content_json = json.loads(content)
    else:
        content_json = content
    return content_json


class PlanInputMessage(BaseModel):
    task: str
    model_responses: List
    convs: Optional[List] = None


class JudgeInputMessage(BaseModel):
    task: str
    model_responses: List
    dimension_plan: dict
    convs: Optional[List] = None
    example_paths: Optional[List] = None



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
            print(result_json)
            try:
                score = float(result_json.get("judgement", 0))
            except Exception:
                score = 0.0
            total_score += weight * score
            weight_sum += weight
            detailed_scores[name] = {"weight": weight, "score": score}

        final_score = total_score / weight_sum if weight_sum > 0 else 0.0
        return final_score

    elif eval_mode == "pairwise":
        response_scores = {"response 1": 0, "response 2": 0}
        response_map = {"response 1": "response 1", "response a": "response 1", "response 2": "response 2", "response b": "response 2"}
        weight_sum = 0.0

        for dim in dims:
            name = dim["name"]
            weight = float(dim.get("weight", 0))
            result_json = judge_results.get(name, "{}")
            print(result_json)
            try:
                judgement = result_json.get("judgement", "").strip().lower()
            except Exception:
                judgement = ""
            print(f'Dimension: {name}, Judgement: {judgement}')
            if judgement == "tie":
                response_scores["response 1"] += weight / 2
                response_scores["response 2"] += weight / 2
            elif judgement in ['response 1', 'response 2', 'response a', 'response b']:
                response_scores[response_map.get(judgement)] += weight
            else:
                pass

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

    evaluation_plan, judge_results, final_judgement = None, None, None

    # Initialize PlannerAgent
    planner = PlannerAgent(name='planner', mode='plan')
    plan_user_message = StructuredMessage[PlanInputMessage](
        source="user",
        content=PlanInputMessage(
            task=task,
            model_responses=model_responses,
            convs=convs
        )
    )
    for i in range(3):
        try:
            planner_response = await planner.on_messages([plan_user_message], CancellationToken())
            evaluation_plan = safe_load_json(planner_response.chat_message.content)
            break
        except Exception as e:
            print(f'Warning: generate plan fail! Exception: {e}. Try Again!')
            continue
    # If enable critic, then start planner-critic group chat
    # If not enable critic, then parse planner response and assign judges
    if with_critic:
        pass
    if not evaluation_plan:
        return evaluation_plan, judge_results, final_judgement
    
    try:
        dimensions = evaluation_plan.get("evaluation_dimensions")
        dimension_names = [d['name'] for d in dimensions]
        print(f"--------------------Evaluate {len(dimensions)} Dimensions--------------------")
        print(f"Dimensions are: {dimension_names}")

        # Initialize JudgeAgent
        judge = JudgeAgent(name='judge')
        judge_results = {k['name']:None for k in dimensions}
        # Judge each dimension independently
        for dimension_plan in dimensions:
            dimension_name = dimension_plan['name']
            judge_user_message = StructuredMessage[JudgeInputMessage](
                source="user",
                content=JudgeInputMessage(
                    task=task,
                    model_responses=model_responses,
                    dimension_plan=dimension_plan,
                    convs=convs,
                    example_paths=None
                )
            )
            judge_response = await judge.on_messages([judge_user_message], CancellationToken())
            judge_results[dimension_name] = safe_load_json(judge_response.chat_message.content)
        # If enable judge chat, then start judges' group chat
        # If not enable judge chat, then aggreate judge results to final result
        if judge_chat:
            pass
        print(f"--------------------Aggregate {len(judge_results)} Results--------------------")
        final_judgement = aggregate_final_result(evaluation_plan, judge_results)
    except Exception as e:
        print(f'Evaluation Fail! Exception: {e}')
    print('--------------------End of Evaluation--------------------')
    return evaluation_plan, judge_results, final_judgement


if __name__ == "__main__":
    import asyncio
    import os
    # os.environ["CUDA_VISIBLE_DEVICES"] = "3"



    # evaluate one sample
    task = "Write a summary for the given context.\nContext: Artificial intelligence (AI) is transforming many industries, from healthcare to finance. In healthcare, AI assists doctors in diagnosing diseases faster and more accurately. In finance, AI algorithms detect fraudulent transactions and automate trading. However, experts warn that while AI brings efficiency, it also raises concerns about privacy, job loss, and bias in automated systems. Governments and organizations are now focusing on developing ethical guidelines and regulations for responsible AI use."
    model_responses = [
        "AI is changing industries such as healthcare and finance by improving diagnosis and detecting fraud. Yet, it also causes privacy and job concerns. Governments are working on responsible AI regulations.",
        "AI helps many industries. It can be dangerous and might replace jobs. Healthcare and finance are some examples, and people want laws for AI."
    ]
    convs = None
    asyncio.run(run_pipeline(
        task, 
        model_responses, 
        convs=convs,
        with_critic=False,
        judge_chat=False
    ))

