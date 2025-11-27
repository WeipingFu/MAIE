from .utils import load_json
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer,
)
import torch

# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = '1,2'

client_config = load_json("./config.json").get("client-new")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = 'cpu'
print('Device: {}'.format(device))

class LLamaModelClient:
    def __init__(self):
        print(f"--------------------Load Base Model: {client_config.get('model_name')}--------------------")
        self.tokenizer = AutoTokenizer.from_pretrained(client_config.get('tokenizer_path'))
        self.tokenizer.add_special_tokens({"pad_token": "<PAD>",})
        self.model = AutoModelForCausalLM.from_pretrained(
            client_config.get("model_path"),
            torch_dtype=torch.float16,
            dtype="auto"
        ).to(device)

    async def call(self, messages, thinking=False, max_new_tokens=512):
        # terminators = [self.tokenizer.eos_token_id, self.tokenizer.convert_tokens_to_ids("<|eot_id|>")]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=thinking
        )
        model_inputs = self.tokenizer([text], return_tensors="pt").to(device)
        generated_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=max_new_tokens,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        output_text = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return output_text
    


user_client = LLamaModelClient()