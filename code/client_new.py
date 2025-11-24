from .utils import load_json
# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = '1,2'
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer,
)
import torch

client_config = load_json("./config.json").get("client-new")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = 'cpu'
print('Device: {}'.format(device))

class LLamaModelClient:
    def __init__(self):
        print(f"--------------------Load Base Model: {client_config.get('model_name')}--------------------")
        self.tokenizer = AutoTokenizer.from_pretrained(client_config.get('tokenizer_path'))
        self.tokenizer.add_special_tokens({"pad_token": "<PAD>",})
        self.model = AutoModelForCausalLM.from_pretrained(
            client_config.get("model_path"),
            dtype="auto"
        ).to(device)

    async def call(self, system_prompt, user_prompt, max_new_tokens=512):
        messages = [
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': user_prompt}
        ]
        terminators = [self.tokenizer.eos_token_id, self.tokenizer.convert_tokens_to_ids("<|eot_id|>")]
        input_ids = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(device)
        outputs = self.model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            eos_token_id=terminators,
            pad_token_id=self.tokenizer.pad_token_id,
            do_sample=True,
        )
        output_text = self.tokenizer.decode(outputs[0][input_ids.shape[-1]:], skip_special_tokens=True)
        return output_text
    


user_client = LLamaModelClient()