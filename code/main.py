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
import asyncio


class PlanInputMessage(BaseModel):
    task: str
    model_responses: List
    eval_mode: Optional[str] = None
    convs: Optional[List] = None
    original_evaluation_plan: Optional[str] = None
    feedback: Optional[str] = None

class JudgeInputMessage(BaseModel):
    task: str
    model_responses: List
    dimension_plan: dict
    eval_mode: Optional[str] = None
    convs: Optional[List] = None
    example_paths: Optional[List] = None

class CriticInputMessage(BaseModel):
    task: str
    model_responses: List
    evaluation_plan: str


# aggregate judge results
def aggregate_final_result(plan_json, judge_results, eval_mode=None):
    """
    Aggregate final results based on evaluation mode.
    - If mode == "pointwise": weighted average of scores.
    - If mode == "pairwise": compute response-wise win rates.
    """

    dims = plan_json.get("evaluation_dimensions", [])
    if not eval_mode:
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
            weight = float(dim.get("weight", 0.2))
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

async def plan_critic_collaboration(task, model_responses, eval_mode, convs, original_evaluation_plan, critic_round):
    print('--------------------Start Planner-Critic Collaboration--------------------')
    planner = PlannerAgent(name='reviser', mode='revise')
    critic = CriticAgent(name="critic")
    i = 0
    evaluation_plan_str = json.dumps(original_evaluation_plan)
    # try until exced critic_round
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
        # critic accept the plan, then stop
        decision = critic_feedback.get("decision", '')
        print(f'Decision of CriticAgent: {decision}')
        if decision == 'accept':
            break
        plan_user_message = StructuredMessage[PlanInputMessage](
            source="user",
            content=PlanInputMessage(
                task=task,
                model_responses=model_responses,
                eval_mode=eval_mode,
                convs=convs,
                original_evaluation_plan=evaluation_plan_str,
                feedback=critic_response.chat_message.content
            )
        )
        planner_response = await planner.on_messages([plan_user_message], CancellationToken())
        evaluation_plan_str = planner_response.chat_message.content
        i += 1
    return safe_load_json(evaluation_plan_str), i

async def run_one_judge(judge, task, model_responses, dimension_plan, eval_mode, convs):
    dimension_name = dimension_plan['name']
    judge_user_message = StructuredMessage[JudgeInputMessage](
        source="user",
        content=JudgeInputMessage(
            task=task,
            model_responses=model_responses,
            dimension_plan=dimension_plan,
            eval_mode=eval_mode,
            convs=convs,
            example_paths=None
        )
    )
    judge_response = await judge.on_messages([judge_user_message], CancellationToken())
    one_judge_result = safe_load_json(judge_response.chat_message.content)
    return dimension_name, one_judge_result

async def run_pipeline(task, model_responses, eval_mode=None, convs=None, critic_round=0, judge_chat=True):
    print("--------------------Start Evaluation--------------------")

    if not eval_mode:
        if len(model_responses) == 1:
            eval_mode = 'pointwise'
        elif len(model_responses) == 2:
            eval_mode = 'pairwise'
        else:
            raise ValueError('The count of model_responses = {}, which is not supported!'.format(len(model_responses)))
        
    print(f"Evaluation Mode: {eval_mode}, Critic Round: {critic_round}, Multi-round Judge: {judge_chat}")

    evaluation_plan, judge_results, final_judgement = None, None, None
    revise_count = -1

    # Initialize PlannerAgent
    planner = PlannerAgent(name='planner', mode='plan')
    plan_user_message = StructuredMessage[PlanInputMessage](
        source="user",
        content=PlanInputMessage(
            task=task,
            model_responses=model_responses,
            eval_mode=eval_mode,
            convs=convs
        )
    )
    # Frist plan
    planner_response = await planner.on_messages([plan_user_message], CancellationToken())
    original_evaluation_plan = safe_load_json(planner_response.chat_message.content)

    if not original_evaluation_plan or len(original_evaluation_plan) == 0:
        return None, None, None, revise_count

    # If enable critic, then start planner-critic collaboration
    if critic_round > 0:
        evaluation_plan, revise_count = await plan_critic_collaboration( 
            task=task, 
            model_responses=model_responses,
            eval_mode=eval_mode,
            convs=convs, 
            original_evaluation_plan=original_evaluation_plan, 
            critic_round=critic_round
        )
    else:
        evaluation_plan = copy.deepcopy(original_evaluation_plan)
    # print(f'Evaluation Plan\n{evaluation_plan}')

    if not evaluation_plan or len(evaluation_plan) == 0:
        return None, None, None, revise_count
    
    # Parse planner response and assign judges
    try:
        dimensions = evaluation_plan.get("evaluation_dimensions")
        dimension_names = [d['name'] for d in dimensions]
        print(f"--------------------Evaluate {len(dimensions)} Dimensions--------------------")
        print(f"Dimensions are: {dimension_names}")

        # Initialize JudgeAgent
        judge = JudgeAgent(name='judge')
        judge_results = {k['name']:None for k in dimensions}
        # Judge each dimension independently
        judge_tasks = [
            run_one_judge(judge, task, model_responses, dimension_plan, eval_mode, convs) 
            for dimension_plan in dimensions
        ]
        judge_outputs = await asyncio.gather(*judge_tasks, return_exceptions=True)
        for dimension_name, one_judge_result in judge_outputs:
            if one_judge_result and len(one_judge_result) > 0:
                judge_results[dimension_name] = one_judge_result
    
        # If enable judge chat, then start judges' group chat
        # If not enable judge chat, then aggreate judge results to final result
        if judge_chat:
            pass
        print(f"--------------------Aggregate {len(judge_results)} Results--------------------")
        final_judgement = aggregate_final_result(evaluation_plan, judge_results, eval_mode)
    except Exception as e:
        print(f'Evaluation Fail! Exception: {e}')
    print('--------------------End of Evaluation--------------------')
    return evaluation_plan, judge_results, final_judgement, revise_count


if __name__ == "__main__":
    
    # evaluate one sample
    task = "Write a summary for the given context.\nContext: Artificial intelligence (AI) is transforming many industries, from healthcare to finance. In healthcare, AI assists doctors in diagnosing diseases faster and more accurately. In finance, AI algorithms detect fraudulent transactions and automate trading. However, experts warn that while AI brings efficiency, it also raises concerns about privacy, job loss, and bias in automated systems. Governments and organizations are now focusing on developing ethical guidelines and regulations for responsible AI use."
    model_responses = [
        "AI is changing industries such as healthcare and finance by improving diagnosis and detecting fraud. Yet, it also causes privacy and job concerns. Governments are working on responsible AI regulations.",
        "AI helps many industries. It can be dangerous and might replace jobs. Healthcare and finance are some examples, and people want laws for AI."
    ]
    convs = None
    evaluation_plan, judge_results, final_judgement, revise_count = asyncio.run(run_pipeline(
        task, 
        model_responses, 
        eval_mode='pairwise',
        convs=convs,
        critic_round=3,
        judge_chat=False
    ))
    print(f'Evaluation Plan\n{evaluation_plan}')
    print(f'Judge Results\n{judge_results}')
    print(f'Final Judgement: {final_judgement}')
    print(f'Revise Count: {revise_count}')
  