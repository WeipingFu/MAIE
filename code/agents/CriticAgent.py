from typing import List, Sequence, Literal, Optional
from autogen_agentchat.agents import BaseChatAgent
from autogen_agentchat.base import Response
from autogen_core.model_context import UnboundedChatCompletionContext
from autogen_agentchat.messages import BaseChatMessage, TextMessage
from autogen_core import CancellationToken
from pydantic import BaseModel, Field

from ..utils import load_json, read_text
from ..client import client_config, user_client

critic_config = load_json("./config.json").get("critic")


# --- Format of Critic Response ---
class CriticResponse(BaseModel):
    """
    Structured response returned by the Critic Agent.
    """
    decision: Literal["accept", "revise"] = Field(description="The decision Critic Agent make.")
    confidence: float = Field(description="The confidence of Critic Agent's decision.")
    issues: Optional[List[str]] = Field(default_factory=list, description="Optional list of issues of the evaluation plan. If there are no issues, then return a blank list.")
    suggestion: str = Field(description="if the decision is 'revise', return suggestions for revision; if the decision is 'accept', return empty string.")
    rationale: str = Field(description="The rationale for this decision.")


# --- User prompt for Critic ---
class UserPrompt:
    def __init__(self):
        self._user_prompt = read_text(critic_config.get("prompt_path", "../prompts/critic.txt"))


class CriticAgent(BaseChatAgent):
    def __init__(
        self,
        name: str = "critic",
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
        # runtime_payload = messages[-1].content
        # task = runtime_payload["task"]
        # model_responses = runtime_payload["model_responses"]
        # convs = runtime_payload["convs"]
        # content = self._user_prompt.generate_user_prompt(task, model_responses, None, convs)
        # result = await self._model_client.create(
        #     [
        #         SystemMessage(content=self._system_message),
        #         UserMessage(content=content)
        #     ],
        #     json_output=CriticResponse
        # )
        # response_message = ChatMessage(content=result.content, source=self.name)
        # return Response(chat_message=response_message)

    async def on_reset(self, cancellation_token: CancellationToken) -> None:
        pass
        # await self._model_context.clear()