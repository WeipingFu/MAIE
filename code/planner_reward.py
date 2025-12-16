from swift.llm.rl.reward import RewardFn
from .agents.CriticAgent import CriticAgent
from .agents.JudgeAgent import JudgeAgent
from autogen_agentchat.messages import StructuredMessage
from .main import CriticInputMessage, JudgeInputMessage
from autogen_core import CancellationToken
from .utils import safe_load_json



class PlannerReward(RewardFn):
    def __init__(self):
        self.critic = CriticAgent(name="Critic")
        self.judge = JudgeAgent(name='Judge', mode='judge')


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
        critic_response = await self.critic.on_messages([critic_user_message], CancellationToken())
        critic_feedback = safe_load_json(critic_response.chat_message.content)
        decision = critic_feedback.get("decision", '')
        return decision == 'accept'


    async def run_one_judge(self, dimension_plan, task, model_responses, first_judgement, dependency_result_dict, eval_mode, convs):
        dimension_name = dimension_plan['name']
        judge_user_message = StructuredMessage[JudgeInputMessage](
            source="user",
            content=JudgeInputMessage(
                task=task,
                model_responses=model_responses,
                dimension_plan=dimension_plan,
                first_judgement=first_judgement,
                dependency_result_dict=dependency_result_dict,
                eval_mode=eval_mode,
                convs=convs,
                example_paths=None
            )
        )
        judge_response = await self.judge.on_messages([judge_user_message], CancellationToken())
        one_judge_result = safe_load_json(judge_response.chat_message.content)
        return dimension_name, one_judge_result
    

    async def __call__(self, completions, **kwargs):
        """
        completions: RL rollout output
        kwargs - dataset columns:
            - vanilla_prompt
            - label (ground-truth)
            - task
            - model_response
            - ...
        """
        evaluation_plan_str = completions["text"]
        evaluation_plan = safe_load_json(evaluation_plan_str)
        vanilla = kwargs["vanilla_prompt"]
        label = kwargs["label"]
        task = kwargs["task"]
        model_responses = kwargs["model_responses"]
        eval_mode = kwargs["eval_mode"]
        convs = kwargs["convs"]
      
        # Step 1: critic
        critic_accept = await self.run_critic(evaluation_plan_str, task, model_responses, eval_mode, convs)

        reward = 0
        if not critic_accept:
            reward -= 1

        # Step 2: judge
        dimensions = evaluation_plan.get("evaluation_dimensions")
        dimension_names = [d['name'] for d in dimensions]
        judge_tasks = [
            self.run_one_judge(dimension_plan, task, model_responses, None, None, eval_mode, convs)
            for dimension_plan in dimensions
        ]
        judge_pred = self.run_judge(plan, task, response)
        judge_correct = (judge_pred == label)

        vanilla_correct = (vanilla == label)

        if judge_correct:
            if vanilla_correct:
                reward += 1
            else:
                reward += 2
        else:
            if vanilla_correct:
                reward -= 2
            else:
                reward -= 1

        return [reward]
