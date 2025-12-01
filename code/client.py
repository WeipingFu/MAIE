from .utils import load_json
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer,
)
import torch
from vllm import LLM, SamplingParams
from openai import OpenAI
import asyncio

import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

client_config = load_json("config.json").get("client-new")
client_vllm_config = load_json("config.json").get("client-vllm")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = 'cpu'
print('Device: {}'.format(device))

class ModelClient:
    def __init__(self):
        print(f"--------------------Load Base Model: {client_config.get('model_name')}--------------------")
        self.tokenizer = AutoTokenizer.from_pretrained(client_config.get('tokenizer_path'))
        self.tokenizer.add_special_tokens({"pad_token": "<PAD>",})
        self.model = AutoModelForCausalLM.from_pretrained(
            client_config.get("model_path"),
            dtype=torch.bfloat16
        ).to(device)

    async def call(self, messages, thinking=False, max_new_tokens=256):
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
            do_sample=False,
            max_new_tokens=max_new_tokens,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        output_text = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return output_text
    

class ModelCientVLLM:
    def __init__(self, mode='offline'):
        print(f"--------------------Call Mode: {mode}, Load Base Model: {client_config.get('model_name')}--------------------")
        self.mode = mode
        if mode == 'offline':
            self.llm = LLM(
                model=client_vllm_config.get("model_path"),
                tokenizer=client_vllm_config.get("tokenizer_path"),
                dtype="bfloat16"
            )
            self.tokenizer = self.llm.get_tokenizer()

        elif mode == 'online':
            self.llm = OpenAI(
                base_url=client_vllm_config.get("base_url"),
                api_key=client_vllm_config.get("api_key")
            )

        else:
            raise ValueError(f"Unknown mode: {mode}, must be 'offline' or 'online'")
    

    # offline vLLM
    async def call_offline(self, messages, temperature=0.0, max_new_tokens=256):
        try:
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False
            )
            sampling_params = SamplingParams(
                temperature=temperature,
                max_tokens=max_new_tokens,
            )
            outputs = self.llm.generate([prompt], sampling_params)
            return outputs[0].outputs[0].text.strip()
        
        except Exception as e:
            print("[VLLM-OFFLINE] Call failed:", e)
            return ""
    

    # online vLLM / OpenAI
    async def call_online(self, messages, temperature=0.0, max_new_tokens=256):
        try:
            response = self.llm.chat.completions.create(
                model=client_vllm_config.get("model_name"),
                messages=messages,
                temperature=temperature,
                max_tokens=max_new_tokens,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print("[VLLM-ONLINE] Call failed:", e)
            return ""


    # unified call
    async def call(self, messages, temperature=0.0, max_new_tokens=256):
        if self.mode == 'offline':
            return await self.call_offline(
                messages,
                temperature=temperature,
                max_new_tokens=max_new_tokens
            )

        elif self.mode == 'online':
            return await self.call_online(
                messages,
                temperature=temperature,
                max_new_tokens=max_new_tokens
            )

        else:
            raise RuntimeError(f"Invalid mode state: {self.mode}")


    # call batch
    async def call_offline_batch(self, messages_list, temperature=0.0, max_new_tokens=256):
        try:
            prompts = [
                self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False
                )
                for messages in messages_list
            ]
            sampling_params = SamplingParams(
                temperature=temperature,
                max_tokens=max_new_tokens,
            )
            
            outputs = self.llm.generate(prompts, sampling_params)
            results = []
            for out in outputs:
                if not out.outputs:
                    results.append("")
                else:
                    results.append(out.outputs[0].text.strip())
        except Exception as e:
            print("[VLLM-OFFLINE-BATCH] generate failed:", e)
            return [""] * len(prompts)
        
        return results


# user_client = ModelClient()
user_client = ModelCientVLLM(mode='offline')