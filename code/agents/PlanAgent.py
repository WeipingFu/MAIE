from typing import List, Sequence, Literal, Optional
from autogen_agentchat.agents import BaseChatAgent
from autogen_agentchat.base import Response
from autogen_agentchat.messages import BaseChatMessage, TextMessage
from autogen_core import CancellationToken
from pydantic import BaseModel, Field
from jinja2 import Template

from ..utils import load_json, read_text, clean_json
# from ..client_llamacpp import client_config, user_client
from ..client import user_client
# from ..client import ModelCientVLLM

planner_config = load_json("./config.json").get("planner")



# --- JudgeAgent Details ---
class AgentDetails(BaseModel):
    """Defines the role, task, and specific steps for the evaluation Agent."""
    role_name: str = Field(description="The name of the JudgeAgent.")
    role_description: str = Field(description="The role the Agent plays and its primary responsibilities.")
    evaluation_task: str = Field(description="A concise description of the evaluation task the Agent must judge.")
    evaluation_steps: List[str] = Field(description="Detailed, step-by-step guidelines that the Agent must follow.")

# --- Evaluation Dimension ---
class EvaluationDimension(BaseModel):
    """Defines a single dimension within a multi-dimensional evaluation system."""
    name: str = Field(description="The name of the dimension.")
    definition: str = Field(description="A definition describing what this dimension measures.")
    rationale: str = Field(description="Explanation of why this dimension is important for the evaluation.")
    weight: float = Field(description="The weight of this dimension in the final total score (0.0-1.0). The sum of all weights should be 1.")
    evaluation_granularity: str = Field(
        description='The scope and granularity of the evaluation: "holistic" (judges the response as a whole), "localized" (focuses on specific portions), "stepwise" (examines each reasoning or generation step).'
    )
    scoring_scale: str = Field(description='The scoring system used by the evaluator, e.g., "1-5", "0-100", or "binary preference".')
    high_score_indicator: str = Field(description="Describes the characteristics of a high score (full marks).")
    low_score_indicator: str = Field(description="Describes the characteristics of a low score (minimum marks).")
    dependencies: Optional[List[str]] = Field(default_factory=list, description="Optional list of names of other dimensions this dimension related with.")
    assigned_agent: AgentDetails = Field(description="The configuration for the JuegeAgent assigned to perform this dimension's evaluation.")

# --- Format of Planner Response ---
class PlannerResponse(BaseModel):
    """
    Structured response returned by the Planner Agent, used to guide subsequent evaluation flows.
    """
    task_type: str = Field(description="Identifies the type of task, e.g., 'Code Generation', 'Summarization'.")
    sub_task: str = Field(description="The specific sub-task.")
    evaluation_mode: Literal["pointwise", "pairwise"] = Field(
        description="Specifies whether the evaluation targets a single response (pointwise) or compares two responses (pairwise)."
    )
    evaluation_dimensions: List[EvaluationDimension] = Field(
        description="A list of multi-dimensional criteria to be used for evaluating the generation result."
    )

# --- User prompt for Planner ---
class UserPrompt:
    def __init__(self, mode='plan'):
        self._mode = mode
        if self._mode == 'revise':
            self._user_prompt = read_text(planner_config.get("revise_prompt_path", "../prompts/plan-revise.txt"))
        else:
            self._user_prompt = read_text(planner_config.get("plan_prompt_path", "../prompts/plan.txt"))

    def get_conv_history(self, convs):
        history_str = ''
        if convs and len(convs) > 0:
            history_str = '[Conversation History Between User and Models]\n'
            for idx, conv in enumerate(convs):
                history_str += '[Turn {}]\n[User]\n{}\n[Model a]\n{}\n[Model b]\n{}\n\n'.format(str(idx+1), conv['user'], conv['a'], conv['b'])
        return history_str
    
    def get_user_criteria(self, eval_mode, criteria_list=None):
        user_criteria = ''
        scoring_scale = 'binary preference' if eval_mode == 'pairwise' else '1-5'
        if criteria_list and len(criteria_list) > 0:
            # [{"dimension":"", "description":"", "scoring_scale":"", "high_score_indicator":"", "low_score_indicator":""}, ...]
            user_criteria += '[User-defined Evaluation Criteria]\n'
            for i, item in enumerate(criteria_list):
                dimension_str = item.get('dimension','')+"-"+item.get('description')
                one = f"Dimension: {dimension_str}\nScoring Scale: {item.get('scoring_scale', scoring_scale)}\nHigh Score Indicator: {item.get('high_score_indicator','')}\nLow Score Indicator: {item.get('low_score_indicator','')}\n"
                user_criteria += one
        return user_criteria
    
    def get_examplestr(self):
        example_str = ''
        example_paths = planner_config.get('example_paths', None)
        if example_paths:
            examples = []
            for path in example_paths:
                examples.append(read_text(path))
            example_str = '\n\n'.join(['[Start of Example {}]\n{}\n[End of Example {}]'.format(i+1, ex, i+1) for i,ex in enumerate(examples)]) + '-'*40
        return example_str
    
    def get_model_response(self, model_responses):
        mapping = {i: chr(ord('a') + i - 1) for i in range(1, 27)}
        if len(model_responses) == 1:
            model_response_str = model_responses[0]
        else:
            model_response_str = '\n'.join([f'[Response of Model {mapping[idx+1]}]\n{response}' for idx, response in enumerate(model_responses)])
        return model_response_str

    def generate_user_prompt(self, task, model_responses, eval_mode=None, original_evaluation_plan=None, feedback=None, convs=None, criteria_list=None):
        if not eval_mode:
            if len(model_responses) == 1:
                eval_mode = 'pointwise'
            elif len(model_responses) == 2:
                eval_mode = 'pairwise'
            else:
                raise ValueError('The count of model_responses = {}, which is not supported!'.format(len(model_responses)))
        template_vars = {
            "examples": self.get_examplestr(),
            "history": self.get_conv_history(convs),
            "task_description": task,
            "model_response": self.get_model_response(model_responses),
            "criteria": self.get_user_criteria(eval_mode, criteria_list)    
        }
        if self._mode == 'revise':
            template_vars["original_evaluation_plan"] = original_evaluation_plan
            template_vars["feedback"] = feedback
        else:
            template_vars["evaluation_mode"] = eval_mode
        template = Template(self._user_prompt)
        content = template.render(**template_vars)
        # print(f'User Prompt:\n{content}')
        return content


class PlannerAgent(BaseChatAgent):
    def __init__(
        self,
        name: str = "Planner",
        description: str = "An agent that generate the evaluation plan.",
        model: str = planner_config.get("model_name", "plan_model"),
        mode: str = "plan"                          # plan or revise
    ):
        super().__init__(name=name, description=description)
        # self._model_context = UnboundedChatCompletionContext()
        self._model_client = user_client
        self._system_message = "You are the Planning Agent in a multi-agent system. Your function is to design a detailed, structured evaluation plan for the given instance input. If the input contains revision feedback, revise the existing plan accordingly."
        self._model = model
        self._mode = mode
        self._user_prompt = UserPrompt(mode)
        
    @property
    def produced_message_types(self) -> Sequence[type[BaseChatMessage]]:
        return (TextMessage,)

    async def on_messages(self, messages: Sequence[BaseChatMessage], cancellation_token: CancellationToken) -> Response:
        print(f"PlannerAgent is working, mode is {self._mode} ...")
        runtime_payload = messages[-1].content
        task = runtime_payload.task
        model_responses = runtime_payload.model_responses
        eval_mode = runtime_payload.eval_mode
        convs = runtime_payload.convs
        original_evaluation_plan = runtime_payload.original_evaluation_plan
        feedback = runtime_payload.feedback
        criteria_list = runtime_payload.criteria_list
        content = self._user_prompt.generate_user_prompt(
            task, 
            model_responses, 
            eval_mode=eval_mode,
            original_evaluation_plan=original_evaluation_plan,
            feedback=feedback, 
            convs=convs,
            criteria_list=criteria_list
        )

        # Call the LLM
        model_messages = [
            {'role': 'system', 'content': self._system_message},
            {'role': 'user', 'content': content}
        ]
        result = await self._model_client.call(
            model_messages, 
            lora_path=planner_config.get("lora_path", ""),
            temperature=planner_config.get("temperature", 0.0),
            max_new_tokens=planner_config.get("max_new_tokens"))
        # print(result)
        # validate LLM result
        result = clean_json(result)
        clean_json_str = '{}'
        print(f'{self.name} Result: {result}\n')
        try:
            parsed = PlannerResponse.model_validate_json(result)
            clean_json_str = parsed.model_dump_json(indent=2)
        except Exception as e:
            print("Warning: PlannerResponse validation failed, return empty json str. Exception:", e)
            
        response_message = TextMessage(content=clean_json_str, source=self.name)
        return Response(chat_message=response_message)

    async def on_reset(self, cancellation_token: CancellationToken) -> None:
        pass
        # await self._model_context.clear()