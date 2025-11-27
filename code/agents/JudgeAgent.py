from typing import List, Sequence, Optional
from autogen_agentchat.agents import BaseChatAgent
from autogen_agentchat.base import Response
from autogen_agentchat.messages import BaseChatMessage, TextMessage
from autogen_core import CancellationToken
from pydantic import BaseModel, Field
from jinja2 import Template
from ..utils import load_json, read_text, safe_load_json
# from ..client_llamacpp import client_config, user_client
from ..client import client_config, user_client

judge_config = load_json("./config.json").get("judge")


# --- Format of Judge Response ---
class JudgeResponse(BaseModel):
    """
    Structured response returned by the Judge Agent.
    """
    dimension: str = Field(description="The specific dimension that the Agent judge.")
    judgement: str = Field(description="The judge result of current dimension and input.")
    step_by_step_evaluation_process: str = Field(description="Detailed step-by-step process during evaluation.")
    evidences: Optional[List[str]] = Field(default_factory=list, description="List of evidences you found in the context.")
    confidence: float = Field(description="The confidence of the Agent's judgement (0.0-1.0).")


# --- User prompt for Judge ---
class UserPrompt:
    def __init__(self):
        self._user_prompt = read_text(judge_config.get("prompt_path", "../prompts/judge.txt"))

    def get_conv_history(self, convs):
        history_str = ''
        if convs and len(convs) > 0:
            history_str = '[Conversation History Between User and Models]\n'
            for idx, conv in enumerate(convs):
                history_str += '[Turn {}]\nUser: {}\nModel A: {}\nModel B: {}\n'.format(str(idx+1), conv['user'], conv['a'], conv['b'])
            history_str += '\n\n\n'
        return history_str
    
    def get_examplestr(self, example_paths):
        if example_paths:
            examples = []
            for path in example_paths:
                examples.append(read_text(path))
            self.example_str = '\n\n'.join(['[Start of Example {}]\n{}\n[End of Example {}]'.format(i+1, ex, i+1) for i,ex in enumerate(examples)]) + '-'*40

    def generate_user_prompt(self, task, model_responses, dimension_plan, convs=None, example_paths=None):
        if len(model_responses) == 1:
            eval_mode = 'pointwise'
        elif len(model_responses) == 2:
            eval_mode = 'pairwise'
        else:
            raise ValueError('The count of model_responses = {}, which is not supported!'.format(len(model_responses)))
        template_vars = {
            "examples": self.get_examplestr(example_paths),
            "history": self.get_conv_history(convs),
            "generation_task": task,
            "model_response": '\n'.join(['[Response {}]\n{}'.format(i+1, output) for i,output in enumerate(model_responses)]),
            "dimension": dimension_plan.get("name") + ': ' + dimension_plan.get("definition"),
            "evaluation_task": dimension_plan.get("assigned_agent").get("evaluation_task"),
            "evaluation_mode": eval_mode,
            "scoring_scale": dimension_plan.get("scoring_scale"),
            "high_score_indicator": dimension_plan.get("high_score_indicator"),
            "low_score_indicator": dimension_plan.get("low_score_indicator"),
            "granularity": dimension_plan.get("evaluation_granularity"),
            "steps": dimension_plan.get("assigned_agent").get("evaluation_steps")
        }
        template = Template(self._user_prompt)
        content = template.render(**template_vars)
        print(f'User Prompt:\n{content}')
        return content


class JudgeAgent(BaseChatAgent):
    def __init__(
        self,
        name: str = "Judge",
        description: str = "An agent that judge model's response(s) for given task.",
        model: str = client_config.get("model_name", "model")
    ):
        super().__init__(name=name, description=description)
        # self._model_context = UnboundedChatCompletionContext()
        self._model_client = user_client
        self._system_message = "You are a highly specialized Judge Agent. Your task is to evaluate one assigned dimension of the response and perform a structured judgment based on the provided context and the criteria for your specific dimension."
        self._model = model
        self._user_prompt = UserPrompt()
        
    @property
    def produced_message_types(self) -> Sequence[type[BaseChatMessage]]:
        return (TextMessage,)

    async def on_messages(self, messages: Sequence[BaseChatMessage], cancellation_token: CancellationToken) -> Response:
        runtime_payload = messages[-1].content
        task = runtime_payload.task
        model_responses = runtime_payload.model_responses
        dimension_plan = runtime_payload.dimension_plan
        convs = runtime_payload.convs
        example_paths = runtime_payload.example_paths
        # print(dimension_plan)
        print(f"JudgeAgent is working, Dimension is {dimension_plan.get('name')}")
        content = self._user_prompt.generate_user_prompt(task, model_responses, dimension_plan, convs, example_paths)
        # result = await self._model_client.create(
        #     [
        #         SystemMessage(content=self._system_message),
        #         UserMessage(content=content, source='user')
        #     ],
        #     json_output=JudgeResponse
        # )
        # response_message = TextMessage(content=result.content, source=self.name)

        # Call the LLM
        messages = [
            {'role': 'system', 'content': self._system_message},
            {'role': 'user', 'content': content}
        ]
        result = await self._model_client.call(
            messages,
            thinking=judge_config.get("thinking"),
            max_new_tokens=judge_config.get("max_new_tokens"))
        
        # validate LLM result
        result = result.strip().replace('```json','').replace('```','')
        clean_json_str = '{}'
        try:
            parsed = JudgeResponse.model_validate_json(result)
            clean_json_str = parsed.model_dump_json(indent=2)
        except Exception as e:
            print("Warning: JudgeResponse validation failed, return empty json str. Exception:", e)
            
        response_message = TextMessage(content=clean_json_str, source=self.name)
        return Response(chat_message=response_message)

    async def on_reset(self, cancellation_token: CancellationToken) -> None:
        pass
        # await self._model_context.clear()