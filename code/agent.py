from utils import load_json, read_text
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from api_request import completion

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('device: {}'.format(device))

class Agent:
    def __init__(self, arg_path:str):
        self.params = load_json(arg_path)
        self.model_name = self.params.get("model_name", '')
        model_type = self.params.get("model_type", 'open')
        self.role = self.params.get("role", '')
        self.tokenizer = None
        if model_type == 'open':
            self.load_model(self.params.get('tokenizer_path',''), self.params.get('model_path',''))
        self.examples = []
    
    def load_model(self, tokenizer_path:str, model_path:str):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype="auto"
        ).to(device)
    
    def load_promptTemp(self):
        prompt_path = self.params.get("prompt_path", '')
        if not prompt_path:
            raise ValueError('Empty prompt path for agent!')
        self.prompt_template = read_text(prompt_path)

    def get_userprompt(self, task:str) -> None:
        self.load_promptTemp()
        self.user_prompt = self.prompt_template.replace('#task_description', task)
        if self.examples:
            example_str = '\n\n'.join(['[Start of Example {}]\n{}\n[End of Example {}]'.format(i+1, ex, i+1) for i,ex in enumerate(self.examples)])
            self.user_prompt = self.user_prompt.replace('#examples', example_str)

    def get_examples(self, example_paths:list = None) -> None:
        example_paths = self.params.get('example_paths', example_paths)
        if example_paths:
            for path in example_paths:
                self.examples.append(read_text(path))
        
    def get_messages(self, pre_messages:list = None) -> None:
        if not pre_messages:
            self.messages = [
                {'role': 'system', 'content': self.role},
                {'role': 'user', 'content': self.user_prompt}
            ]
        else:
            self.messages = pre_messages + [
                {'role': 'user', 'content': self.user_prompt}
            ]
        
    def get_response(self, max_new_tokens:int=512, thinking:bool=False, prt:bool=False) -> str:
        resp = ''
        max_new_tokens = self.params.get('max_new_tokens', max_new_tokens)
        thinking = self.params.get('thinking', thinking)
        prt = self.params.get("prt", prt)
        if prt:
            print('Messages:')
            print(self.messages)

        if self.tokenizer is None:
            resp = completion(self.model_name, self.messages, prt=prt)
        else:
            text = self.tokenizer.apply_chat_template(
                self.messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=thinking
            )
            model_inputs = self.tokenizer([text], return_tensors="pt").to(device)
            generated_ids = self.model.generate(
                **model_inputs,
                max_new_tokens=max_new_tokens
            )
            generated_ids = [
                output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]
            resp = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
            if prt:
                print('Response: ')
                print(resp)
        return resp
