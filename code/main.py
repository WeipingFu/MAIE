from .agents.PlanAgent import PlannerAgent
from .agents.CriticAgent import CriticAgent
from .agents.JudgeAgent import JudgeAgent
from autogen_agentchat.messages import StructuredMessage
from typing import List, Optional
from pydantic import BaseModel
from autogen_core import CancellationToken
import copy
import json
from .utils import safe_load_json

class PlanInputMessage(BaseModel):
    task: str
    model_responses: List
    convs: Optional[List] = None
    original_evaluation_plan: Optional[str] = None
    feedback: Optional[str] = None

class ReviseInputMessage(BaseModel):
    task: str
    feedback: str
    pre_messages: List

class JudgeInputMessage(BaseModel):
    task: str
    model_responses: List
    dimension_plan: dict
    convs: Optional[List] = None
    example_paths: Optional[List] = None

class CriticInputMessage(BaseModel):
    task: str
    model_responses: List
    evaluation_plan: str


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
            weight = float(dim.get("weight", 0.2))
            result_json = judge_results.get(name, None)
            # print(result_json)
            try:
                score = float(result_json.get("judgement"))
            except Exception:
                score = 0.0
                weight = 0.0
            total_score += weight * score
            weight_sum += weight
            detailed_scores[name] = {"weight": weight, "score": score}

        final_score = total_score / weight_sum if weight_sum > 0 else 9999
        return final_score

    elif eval_mode == "pairwise":
        response_scores = {"model_a": 0, "model_b": 0}
        mapping = {
            "model a": "model_a", "model b": "model_b", 
            "model_a":"model_a", "model_b":"model_b",
            "response a":"model_a", "response b":"model_b", 
            "response 1":"model_a", "response 2":"model_b", 
            "model 1":"model_a", "model 2":"model_b"  
        }
        weight_sum = 0.0

        for dim in dims:
            name = dim["name"]
            weight = float(dim.get("weight"), 0.2)
            result_json = judge_results.get(name, None)
            # print(result_json)
            try:
                judgement = result_json.get("judgement", "").strip().lower()
            except Exception:
                judgement = None
            print(f'Dimension: {name}, Judgement: {judgement}')
            if judgement == "tie":
                response_scores["model_a"] += weight / 2
                response_scores["model_b"] += weight / 2
            elif judgement in list(mapping.keys()):
                response_scores[mapping[judgement]] += weight
            else:
                weight = 0.0

            weight_sum += weight
            detailed_scores[name] = {"weight": weight, "judgement": judgement}
        
        if weight_sum > 0:
            response_scores = {k: v / weight_sum for k, v in response_scores.items()}
        # print(f'response_scores:{response_scores} =====================================================')
        if response_scores["model_a"] == 0 and response_scores["model_b"] == 0:
            final_judgement = None
        elif abs(response_scores["model_a"] - response_scores["model_b"]) < 1e-6:
            final_judgement = "tie"
        elif response_scores["model_a"] > response_scores["model_b"]:
            final_judgement = "model_a"
        else:
            final_judgement = "model_b"
        print(f'Final Judgement: {final_judgement}')
        return final_judgement
    else:
        raise ValueError(f"Unknown evaluation mode: {eval_mode}")

async def plan_critic_collaboration(planner, critic, task, model_responses, convs, original_evaluation_plan, critic_round):
    i = 0
    planner._mode = 'revise'
    evaluation_plan_str = json.dumps(original_evaluation_plan)
    while i < critic_round:
        critic_user_message = StructuredMessage[CriticInputMessage](
            source="user",
            content=CriticInputMessage(
                task=task,
                model_responses=model_responses,
                evaluation_plan=evaluation_plan_str
            )
        )
        critic_response = await critic.on_messages([critic_user_message], CancellationToken())
        critic_feedback = safe_load_json(critic_response.chat_message.content)
        if critic_feedback.get("decision", '') == 'accept':
            break
        plan_user_message = StructuredMessage[PlanInputMessage](
            source="user",
            content=PlanInputMessage(
                task=task,
                model_responses=model_responses,
                convs=convs,
                original_evaluation_plan=evaluation_plan_str,
                feedback=critic_response.chat_message.content
            )
        )
        planner_response = await planner.on_messages([plan_user_message], CancellationToken())
        evaluation_plan_str = planner_response.chat_message.content
        i += 1
    return safe_load_json(evaluation_plan_str)


async def run_pipeline(task, model_responses, convs=None, critic_round=0, judge_chat=True):
    print("--------------------Start Evaluation--------------------")
    print(f"Critic Round: {critic_round}\nMulti-round Judge: {judge_chat}")

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
    # Frist plan
    planner_response = await planner.on_messages([plan_user_message], CancellationToken())
    original_evaluation_plan = safe_load_json(planner_response.chat_message.content)

    # If enable critic, then start planner-critic collaboration
    if critic_round > 0:
        critic = CriticAgent(name="critic")
        evaluation_plan = await plan_critic_collaboration(
            planner=planner, 
            critic=critic, 
            task=task, 
            model_responses=model_responses, 
            convs=convs, 
            original_evaluation_plan=original_evaluation_plan, 
            critic_round=critic_round
        )
    else:
        evaluation_plan = copy.deepcopy(original_evaluation_plan)

    # Parse planner response and assign judges
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
            one_judge_result = safe_load_json(judge_response.chat_message.content)
            if one_judge_result:
                judge_results[dimension_name] = one_judge_result
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
        critic_round=0,
        judge_chat=False
    ))

