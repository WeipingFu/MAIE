# from swift.llm.rl.reward import RewardFn
from swift.plugin import ORM, orms
from .agents.CriticAgent import CriticAgent
from .agents.JudgeAgent import JudgeAgent
from .agents.PlanAgent import PlannerResponse
from autogen_agentchat.messages import StructuredMessage
from .main import CriticInputMessage, JudgeInputMessage, run_judge, aggregate_final_result
from autogen_core import CancellationToken
from .utils import clean_json, safe_load_json
from typing import List



class PlannerReward(ORM):
    def __init__(self):
        self.critic = CriticAgent(name="Critic")
        self.judge = JudgeAgent(name="Judge", mode="judge")
        self.revise_judge = JudgeAgent(name="ReviseJudge", mode="revise")

    def judge_distance(pred, label, eval_mode, scale=4.0):
        try:
            if eval_mode == "pairwise":
                return 0.0 if pred == label else 1.0
            else:       # pointwise
                pred = float(pred)
                label = float(label)
                return min(abs(pred - label) / scale, 1.0)
        except Exception:
            return 1.0

    async def run_critic(self, evaluation_plan_str, task, model_responses, eval_mode, convs):
        critic_user_message = StructuredMessage[CriticInputMessage](
            source="user",
            content=CriticInputMessage(
                task=task,
                model_responses=model_responses,
                evaluation_plan=evaluation_plan_str,
                eval_mode=eval_mode,
                convs=convs,
                criteria_list=None
            )
        )
        critic_response = await self.critic.on_messages(
            [critic_user_message], CancellationToken()
        )
        critic_feedback = safe_load_json(critic_response.chat_message.content)

        # invalid critic feedback
        if not critic_feedback:
            return True
        if 'decision' not in critic_feedback:
            return True
        
        return critic_feedback.get("decision", "") == "accept"


    async def run_judge(self, evaluation_plan, task, model_responses, eval_mode, convs):
        dimensions = evaluation_plan.get("evaluation_dimensions", [])
        dimension_names = [d['name'] for d in dimensions]

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
        judge_results = await run_judge(self.judge, judge_inputs)

        # Add judge chat task if there are dependencies
        revise_judge_inputs = []
        for dimension_plan in dimensions:
            dimension_name = dimension_plan.get('name')
            dependencies = [x for x in dimension_plan.get('dependencies',[]) if x in dimension_names and x != dimension_name]
            dependency_results_dict = {k:judge_results[k] for k in dependencies}
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
            judge_revise_results = await run_judge(self.revise_judge, revise_judge_inputs)
            # Update judge results
            for dimension_name, one_judge_result in judge_revise_results.items():
                if one_judge_result and len(one_judge_result) > 0:
                    judge_results[dimension_name] = one_judge_result
        
        return judge_results


    async def __call__(self, completions, **kwargs):
        """
        completions: List[dict], each dict contains rollout output, e.g. {"text": "..."}
        kwargs: batch-aligned dataset columns
        """
        rewards = []

        batch_size = len(completions)

        for i in range(batch_size):
            completion = completions[i]
            # ---------- Format Check (Early Exit) ----------
            try:
                evaluation_plan_str = clean_json(completion["text"])
                evaluation_plan = PlannerResponse.model_validate_json(evaluation_plan_str)
            except Exception:
                # invalid format, reward = -2
                rewards.append(-2.0)
                continue

            task = kwargs["task"][i]
            label = kwargs["label"][i]
            vanilla = kwargs["vanilla_prompt"][i]
            model_responses = kwargs["model_responses"][i]
            eval_mode = kwargs["eval_mode"][i]
            convs = kwargs["convs"][i]
            reward_weights = kwargs["reward_weights"][i]
            
            # ---------- Step 1: Critic ----------
            critic_accept = await self.run_critic(
                evaluation_plan_str, task, model_responses, eval_mode, convs
            )
            λ_c = reward_weights.get("λ_c", 0.2)
            penalty = -λ_c if not critic_accept else 0.0

            # ---------- Step 2: Judge ----------
            judge_results = await run_judge(evaluation_plan, task, model_responses, eval_mode, convs)
            final_judgement = aggregate_final_result(
                evaluation_plan,
                judge_results,
                eval_mode=eval_mode,
                allow_tie=False,
                target_min=1.0,
                target_max=5.0
            )
            
            # ---------- Reward ----------
            d_judge = self.judge_distance(final_judgement, label, eval_mode)
            d_vanilla = self.judge_distance(vanilla, label, eval_mode)
            delta = d_vanilla - d_judge      
            print(f'Label: {label}, Judgement: {final_judgement}, Vanilla: {vanilla}, Distance_Judge: {d_judge}, Distance_Vanilla: {d_vanilla}, Delta: {delta}')

            alpha = reward_weights.get('alpha', 1.0)      # absolute correctness weight
            beta = reward_weights.get('beta', 1.0)        # relative improvement weight
            R = reward_weights.get('R', 2.0)
            r_abs = 1.0 - 2.0 * d_judge  
            judge_reward = max(-R, min(R, alpha * r_abs + beta * delta))      # [-2, 2]

            rewards.append(judge_reward + penalty)

        return rewards



orms['planner_reward_function'] = PlannerReward