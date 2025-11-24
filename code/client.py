import asyncio
from autogen_core.models import UserMessage
from autogen_ext.models.llama_cpp import LlamaCppChatCompletionClient
from .utils import load_json


client_config = load_json("./config.json").get("client")


user_client = LlamaCppChatCompletionClient(
    model_path=client_config.get("model_path"),
    n_ctx=client_config.get("n_ctx"),
    n_gpu_layers=client_config.get("n_gpu_layers"),
    model_info={
        "json_output": client_config.get("json_output"),
        "function_calling": client_config.get("function_calling"),
        "vision": client_config.get("vision"),
        "family": client_config.get("family"),
        "structured_output": client_config.get("structured_output")
    } 
)


async def main(content):
    result = await user_client.create([UserMessage(content=content, source="user")])
    print(result)


# if __name__ == "__main__":
#     asyncio.run(main("What is the capital of France?"))


# Class LLamaModelClient:
#     def 