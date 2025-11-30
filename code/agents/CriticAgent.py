from typing import List, Sequence, Literal, Optional
from autogen_agentchat.agents import BaseChatAgent
from autogen_agentchat.base import Response
from autogen_agentchat.messages import BaseChatMessage, TextMessage
from autogen_core import CancellationToken
from pydantic import BaseModel, Field
from jinja2 import Template

from ..utils import load_json, read_text, clean_json
from ..client import client_config, user_client

critic_config = load_json("./config.json").get("critic")


# --- Format of Critic Response ---
class CriticResponse(BaseModel):
    """
    Structured response returned by the Critic Agent.
    """
    decision: Literal["accept", "revise"] = Field(description="The decision Critic Agent make.")
    issues: Optional[List[str]] = Field(default_factory=list, description="Optional list of issues of the evaluation plan. If there are no issues, then return a blank list.")
    suggestion: str = Field(description="if decision = revise, return suggestions for revision; if decision = accept, return a short reason for this decision.")


# --- User prompt for Critic ---
class UserPrompt:
    def __init__(self):
        self._user_prompt = read_text(critic_config.get("prompt_path", "../prompts/critic.txt"))
    
    def get_user_criteria(self):
        criteria_path = critic_config.get('criteria_path', '')
        user_criteria = ''
        if criteria_path:
            criteria_list = load_json(criteria_path)['evaluation_dimensions']     # {'evaluation_dimensions':[{"dimension":"", "scoring_scale":"", "high_score_indicator":"", "low_score_indicator":""}, ...]}
            user_criteria += '[User-defined Evaluation Criteria]\n'
            for i, item in enumerate(criteria_list):
                one = 'Dimension: {}\nScoring Scale: {}\nHigh Score Indicator: {}\nLow Score Indicator: {}\n'.format(item.get('dimension',''), item.get('scoring_scale'), item.get('high_score_indicator',''), item.get('low_score_indicator',''))
                user_criteria += one
        return user_criteria
    
    def get_examplestr(self):
        example_str = ''
        example_paths = critic_config.get('example_paths', None)
        if example_paths:
            examples = []
            for path in example_paths:
                examples.append(read_text(path))
            example_str = '\n\n'.join(['[Start of Example {}]\n{}\n[End of Example {}]'.format(i+1, ex, i+1) for i,ex in enumerate(examples)]) + '-'*40
        return example_str
    
    def get_model_response(self, model_responses):
        mapping = {i: chr(ord('a') + i - 1) for i in range(1, 27)}
        model_response_str = '\n'.join([f'[Response of Model {mapping[idx+1]}]\n{response}' for idx, response in enumerate(model_responses)])
        return model_response_str

    def generate_user_prompt(self, task, model_responses, evaluation_plan):
        template_vars = {
            "examples": self.get_examplestr(),
            "task_description": task,
            "model_response": self.get_model_response(model_responses),
            "criteria": self.get_user_criteria(),
            "evaluation_plan": evaluation_plan
        }
        template = Template(self._user_prompt)
        content = template.render(**template_vars)
        # print(f'User Prompt:\n{content}')
        return content

class CriticAgent(BaseChatAgent):
    def __init__(
        self,
        name: str = "Critic",
        description: str = "An agent that review the evaluation plan and provide feedback.",
        model: str = client_config.get("model_name", "model")
    ):
        super().__init__(name=name, description=description)
        # self._model_context = UnboundedChatCompletionContext()
        self._model_client = user_client
        self._system_message = "You are the Plan Critic. Your responsibility is to conduct a thorough review of the provided Evaluation Plan and give feedback."
        self._model = model
        self._user_prompt = UserPrompt()
        
    @property
    def produced_message_types(self) -> Sequence[type[BaseChatMessage]]:
        return (TextMessage,)

    async def on_messages(self, messages: Sequence[BaseChatMessage], cancellation_token: CancellationToken) -> Response:
        print("CriticAgent is working ...")
        runtime_payload = messages[-1].content
        task = runtime_payload.task
        model_responses = runtime_payload.model_responses
        evaluation_plan = runtime_payload.evaluation_plan
        content = self._user_prompt.generate_user_prompt(task, model_responses, evaluation_plan)
        
        # Call the LLM
        messages = [
            {'role': 'system', 'content': self._system_message},
            {'role': 'user', 'content': content}
        ]
        result = await self._model_client.call(
            messages,
            max_new_tokens=critic_config.get("max_new_tokens")
        )
        
        # validate LLM result
        result = clean_json(result)
        clean_json_str = '{}'
        # print(f'{self.name} Result: {result}\n')
        try:
            parsed = CriticResponse.model_validate_json(result)
            clean_json_str = parsed.model_dump_json(indent=2)
        except Exception as e:
            print("Warning: CriticResponse validation failed, return empty json str. Exception:", e)
            
        response_message = TextMessage(content=clean_json_str, source=self.name)
        return Response(chat_message=response_message)

    async def on_reset(self, cancellation_token: CancellationToken) -> None:
        pass
        # await self._model_context.clear()