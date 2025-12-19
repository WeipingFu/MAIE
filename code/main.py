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
    criteria_list: Optional[List] = None

class JudgeInputMessage(BaseModel):
    task: str
    model_responses: List
    dimension_plan: dict
    first_judgement: Optional[dict] = None
    dependency_results_dict: Optional[dict] = None
    eval_mode: Optional[str] = None
    convs: Optional[List] = None
    example_paths: Optional[List] = None

class BatchJudgeInputMessage(BaseModel):
    inputs: List[JudgeInputMessage]

class CriticInputMessage(BaseModel):
    task: str
    model_responses: List
    evaluation_plan: str
    eval_mode: Optional[str] = None
    convs: Optional[List] = None
    criteria_list: Optional[List] = None


# normalize score
def normalize_score(source_min, source_max, target_min, target_max, raw_score):
    if source_min == target_min and source_max == target_max:
        return raw_score
    score = 9999
    target_range = target_max - target_min
    source_range = source_max - source_min
    if source_range > 0 and target_range > 0:
        score = target_min + ((raw_score - source_min) / source_range) * target_range
    return score

# aggregate judge results
def aggregate_final_result(plan_json, judge_results, eval_mode=None, allow_tie=True, target_min=1.0, target_max=5.0):
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
            scoring_scale = dim.get("scoring_scale", "1-5") 
            if scoring_scale == 'binary':
                scoring_scale = '0-1'
            source_min = 1.0
            source_max = 5.0
            try:
                parts = scoring_scale.split('-')
                if len(parts) == 2:
                    source_min = float(parts[0].strip())
                    source_max = float(parts[1].strip())
            except Exception:
                pass

            result_json = judge_results.get(name, None)
            score = 9999
            raw_score = 0.0
            # print(result_json)
            try:
                if result_json and len(result_json) > 0:
                    raw_score = float(result_json.get("judgement"))
                    score = normalize_score(source_min, source_max, target_min, target_max, raw_score)
            except Exception:
                score = 9999
            if score == 9999:
                weight = 0.0
            print(f'Dimension: {name}, Weight: {weight}; Raw score = {raw_score}, Normalized score = {score}')
            total_score += weight * score
            weight_sum += weight
            detailed_scores[name] = {"weight": weight, "score": score}

        final_score = total_score / weight_sum if weight_sum > 0 else 9999
        print(f'Final Score: {final_score}')
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
        valid_dims_info = []

        for dim in dims:
            name = dim["name"]
            weight = float(dim.get("weight", 0.2))
            result_json = judge_results.get(name, None)
            original_weight = weight
            print('result_json', result_json)
            try:
                judgement = None
                if result_json and len(result_json) > 0:
                    judgement = result_json.get("judgement", "").strip().lower()
            except Exception:
                judgement = None
    
            if judgement == "tie":
                response_scores["model_a"] += weight / 2
                response_scores["model_b"] += weight / 2
            elif judgement in mapping:
                response_scores[mapping[judgement]] += weight
            else:
                weight = 0.0
            print(f'Dimension: {name}, Weight: {weight}, Judgement: {judgement}')
            weight_sum += weight
            detailed_scores[name] = {"weight": weight, "judgement": judgement}
            if original_weight > 0 and judgement is not None and judgement != "tie":
                valid_dims_info.append({
                    "name": name, 
                    "weight": original_weight,
                    "judgement": judgement,
                })
        
        if weight_sum > 0:
            response_scores = {k: v / weight_sum for k, v in response_scores.items()}
    
        if response_scores["model_a"] == 0 and response_scores["model_b"] == 0:
            final_judgement = None

        elif abs(response_scores["model_a"] - response_scores["model_b"]) < 1e-6:
            final_judgement = "tie"
            # when tie is not allowed
            if not allow_tie and len(valid_dims_info) > 0:
                max_weight_dim = max(valid_dims_info, key=lambda x: x['weight'])
                if max_weight_dim['judgement'] in mapping:
                    final_judgement = mapping[max_weight_dim['judgement']]

        elif response_scores["model_a"] > response_scores["model_b"]:
            final_judgement = "model_a"

        else:
            final_judgement = "model_b"

        print(f'Final Judgement: {final_judgement}')
        return final_judgement
    
    else:
        raise ValueError(f"Unknown evaluation mode: {eval_mode}")


async def plan_critic_collaboration(task, model_responses, eval_mode, convs, criteria_list, original_evaluation_plan, critic_round):
    print('--------------------Start Planner-Critic Collaboration--------------------')
    planner = PlannerAgent(name='Reviser', mode='revise')
    critic = CriticAgent(name="Critic")
    i = 0
    evaluation_plan_str = json.dumps(original_evaluation_plan)
    # try until exced critic_round
    while i < critic_round:
        critic_user_message = StructuredMessage[CriticInputMessage](
            source="user",
            content=CriticInputMessage(
                task=task,
                model_responses=model_responses,
                evaluation_plan=evaluation_plan_str,
                eval_mode=eval_mode,
                convs=convs,
                criteria_list=criteria_list
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
                criteria_list=criteria_list,
                original_evaluation_plan=evaluation_plan_str,
                feedback=critic_response.chat_message.content
            )
        )
        planner_response = await planner.on_messages([plan_user_message], CancellationToken())
        evaluation_plan_str = planner_response.chat_message.content
        i += 1
    return safe_load_json(evaluation_plan_str), i

async def run_one_judge(judge, task, model_responses, dimension_plan, first_judgement, dependency_results_dict, eval_mode, convs):
    dimension_name = dimension_plan['name']
    is_revise = False
    judge_user_message = StructuredMessage[JudgeInputMessage](
        source="user",
        content=JudgeInputMessage(
            task=task,
            model_responses=model_responses,
            dimension_plan=dimension_plan,
            first_judgement=first_judgement,
            dependency_results_dict=dependency_results_dict,
            eval_mode=eval_mode,
            convs=convs,
            example_paths=None
        )
    )
    judge_response = await judge.on_messages([judge_user_message], CancellationToken())
    one_judge_result = safe_load_json(judge_response.chat_message.content)
    if first_judgement and first_judgement.get('judgement', '') != one_judge_result.get('judgement', ''):
        is_revise = True
    return dimension_name, one_judge_result, is_revise


async def run_judge(judge, inputs):
    judge_user_message = StructuredMessage[BatchJudgeInputMessage](
        source="user",
        content=BatchJudgeInputMessage(inputs=inputs)
    )
    judge_response = await judge.on_messages([judge_user_message], CancellationToken())
    judge_results = safe_load_json(judge_response.chat_message.content)
    for key, item in judge_results.items():
        judge_results[key] = safe_load_json(item)
    return judge_results


async def run_pipeline(task, model_responses, eval_mode=None, convs=None, criteria_list=None, critic_round=0, judge_chat=True, depend='part', allow_tie=True, target_min=1.0, target_max=5.0):
    print("--------------------Start Evaluation--------------------")

    if not eval_mode:
        if len(model_responses) == 1:
            eval_mode = 'pointwise'
        elif len(model_responses) == 2:
            eval_mode = 'pairwise'
        else:
            print('The count of model_responses = {}, which is not supported!'.format(len(model_responses)))
            return None, None, None, -1, -1
        
    print(f"Evaluation Mode: {eval_mode}, Critic Round: {critic_round}, Multi-round Judge: {judge_chat}")

    evaluation_plan, judge_results, final_judgement = None, None, None
    plan_revise_count, judge_revise_count = -1, -1

    # Initialize PlannerAgent
    planner = PlannerAgent(name='Planner', mode='plan')
    plan_user_message = StructuredMessage[PlanInputMessage](
        source="user",
        content=PlanInputMessage(
            task=task,
            model_responses=model_responses,
            eval_mode=eval_mode,
            convs=convs,
            criteria_list=criteria_list
        )
    )
    # 1. Frist plan
    planner_response = await planner.on_messages([plan_user_message], CancellationToken())
    original_evaluation_plan = safe_load_json(planner_response.chat_message.content)

    if not original_evaluation_plan or len(original_evaluation_plan) == 0:
        return None, None, None, plan_revise_count, judge_revise_count
    
    # 2. If enable critic, then start planner-critic collaboration
    if critic_round > 0:
        evaluation_plan, plan_revise_count = await plan_critic_collaboration( 
            task=task, 
            model_responses=model_responses,
            eval_mode=eval_mode,
            convs=convs, 
            criteria_list=criteria_list,
            original_evaluation_plan=original_evaluation_plan, 
            critic_round=critic_round
        )
    else:
        evaluation_plan = copy.deepcopy(original_evaluation_plan)
    # print(f'Evaluation Plan\n{evaluation_plan}')

    if not evaluation_plan or len(evaluation_plan) == 0:
        return None, None, None, plan_revise_count, judge_revise_count
    
    # 3. Parse planner response and assign judges
    # try:
    dimensions = evaluation_plan.get("evaluation_dimensions")
    dimension_names = [d['name'] for d in dimensions]
    print(f"--------------------Evaluate {len(dimensions)} Dimensions--------------------")
    print(f"Dimensions are: {dimension_names}")

    # Initialize JudgeAgent
    judge = JudgeAgent(name='Judge', mode='judge')

    # 3.1. Judge each dimension independently
    judge_inputs = []
    for dimension_plan in dimensions:
        judge_inputs.append(JudgeInputMessage(
            task=task,
            model_responses=model_responses,
            dimension_plan=dimension_plan,
            first_judgement=None,
            dependency_results_dict=None,
            eval_mode=eval_mode,
            convs=convs,
            example_paths=None
        ))
    judge_results = await run_judge(judge, judge_inputs)
    
    # 3.2. If enable judge chat, then start related judges' group chat
    if judge_chat:
        print('--------------------Start Judge Chat--------------------')
        judge_revise_tasks = []
        revise_judge = JudgeAgent(name="ReviseJudge", mode='revise')
        # Add judge chat task if there are dependencies
        # If depend is 'all', all judge chat with each other
        revise_judge_inputs = []
        for dimension_plan in dimensions:
            dimension_name = dimension_plan.get('name')
            if depend == 'all':
                dependencies = [x for x in dimension_names if x != dimension_name]
            else:
                dependencies = [x for x in dimension_plan.get('dependencies',[]) if x in dimension_names and x != dimension_name]
            dependency_results_dict = {k:judge_results[k] for k in dependencies}
            print(f"Dimension: {dimension_name}, Dependencies: {dimension_plan.get('dependencies',[])} {dependencies}")
            if len(dependency_results_dict) > 0:
                revise_judge_inputs.append(JudgeInputMessage(
                    task=task,
                    model_responses=model_responses,
                    dimension_plan=dimension_plan,
                    first_judgement=judge_results[dimension_name],
                    dependency_results_dict=dependency_results_dict,
                    eval_mode=eval_mode,
                    convs=convs,
                    example_paths=None
                ))
                
        # Start judge chat
        if len(revise_judge_inputs) > 0:
            judge_revise_count = 0
            judge_revise_results = await run_judge(revise_judge, revise_judge_inputs)
            # Update judge results
            for dimension_name, one_judge_result in judge_revise_results.items():
                if one_judge_result and len(one_judge_result) > 0:
                    if judge_results[dimension_name].get('judgement') != one_judge_result.get('judgement'):
                        judge_revise_count += 1
                    judge_results[dimension_name] = one_judge_result


    # 3.3. Aggreate judge results to final result
    print(f"--------------------Aggregate {len(judge_results)} Results--------------------")
    final_judgement = aggregate_final_result(evaluation_plan, judge_results, eval_mode, allow_tie, target_min, target_max)
    # except Exception as e:
    #     print(f'Evaluation Fail! Exception: {e}')
    print('--------------------End of Evaluation--------------------')
    return evaluation_plan, judge_results, final_judgement, plan_revise_count, judge_revise_count



def test_one_pairwise():
    # evaluate one sample - pariwise
    task = "Write a summary for the given context.\nContext: Artificial intelligence (AI) is transforming many industries, from healthcare to finance. In healthcare, AI assists doctors in diagnosing diseases faster and more accurately. In finance, AI algorithms detect fraudulent transactions and automate trading. However, experts warn that while AI brings efficiency, it also raises concerns about privacy, job loss, and bias in automated systems. Governments and organizations are now focusing on developing ethical guidelines and regulations for responsible AI use."
    model_responses = [
        "AI is changing industries such as healthcare and finance by improving diagnosis and detecting fraud. Yet, it also causes privacy and job concerns. Governments are working on responsible AI regulations.",
        "AI helps many industries. It can be dangerous and might replace jobs. Healthcare and finance are some examples, and people want laws for AI."
    ]
    convs = None
    evaluation_plan, judge_results, final_judgement, plan_revise_count, judge_revise_count = asyncio.run(run_pipeline(
        task, 
        model_responses, 
        eval_mode='pairwise',
        convs=convs,
        criteria_list=None,
        critic_round=3,
        judge_chat=True,
        allow_tie=False,
        depend='all'
    ))
    print(f'Evaluation Plan\n{evaluation_plan}')
    print(f'Judge Results\n{judge_results}')
    print(f'Final Judgement: {final_judgement}')
    print(f'Plan Revise Count: {plan_revise_count}')
    print(f'Judge Revise Count: {judge_revise_count}')


def test_one_pointwise():
    # evaluate one sample - pointwise
    task = "Imagine a scenario where an individual from the UK is in the United States for a vacation. However, they are struggling to understand the local dialects, accents, and expressions used by the people there. They are also finding it hard to convey their intended message as their phrases and expressions, heavily influenced by their regional factors, are often misunderstood. What steps or strategies can this individual employ to improve their understanding and communication in such a scenario?"
    model_responses = [
        "The individual might find it difficult to adjust to the American dialects and expressions initially, but some strategies might be of slight help. Watching some local American shows can somewhat help in getting accustomed to the local accents. While conversing, try to stick to English that's more generic, it might make communication a little easier. When you don't comprehend what is being said, you might want to ask for explanations, though it might not always help. Sometimes, you might be able to guess the meaning of unfamiliar phrases from the situation or conversation. Active listening might be of little help, but it's worth trying. Using translation apps could be an option, but they might not always translate local expressions accurately."
    ]
    convs = None
    criteria_list = [
        {
            "dimension":"", 
            "description":"Is the model proficient in interpreting and responding to various local dialects, accents, and local expressions? Does it have the ability to comprehend and understand the same expression or phrase used in diverse situations influenced by regional factors?", 
            "scoring_scale":"1-5", 
            "high_score_indicator":"Score 4: The model exhibits a robust understanding of diverse local dialects, accents, and idiomatic expressions, and seldom misreads the context.\nScore 5: The model demonstrates exceptional proficiency in understanding diverse local dialects, accents, and slang, and accurately deciphers the context in all scenarios.", 
            "low_score_indicator":"Score 1: The model displays no comprehension of local dialects, accents, or idioms. It is incapable of understanding the same expression or phrase used in distinct situations influenced by regional factors.\nScore 2: The model exhibits a slight grasp of regional dialects and accents, but often misreads local expressions and context influenced by locality.\nScore 3: The model demonstrates a fair understanding of local dialects, accents, and vernaculars, yet at times misinterprets the context."
        }
    ]
    evaluation_plan, judge_results, final_judgement, plan_revise_count, judge_revise_count = asyncio.run(run_pipeline(
        task, 
        model_responses, 
        eval_mode='pointwise',
        convs=convs,
        criteria_list=criteria_list,
        critic_round=3,
        judge_chat=True
    ))
    # print(f'Evaluation Plan\n{evaluation_plan}')
    # print(f'Judge Results\n{judge_results}')
    # print(f'Final Judgement: {final_judgement}')
    # print(f'Plan Revise Count: {plan_revise_count}')
    # print(f'Judge Revise Count: {judge_revise_count}')


if __name__ == "__main__":
    while True:
        print("Test Sample 1")
        test_one_pairwise()

        print("Test Sample 2")
        test_one_pointwise()