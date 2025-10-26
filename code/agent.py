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
        self.tokenizer = None
        if model_type == 'open':
            self.load_model(self.params.get('tokenizer_path',''), self.params.get('model_path',''))
    
    def load_model(self, tokenizer_path:str, model_path:str):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            dtype="auto"
        ).to(device)
         
    def load_promptTemp(self):
        system_prompt_path = self.params.get("system_prompt_path", '')
        user_prompt_path = self.params.get("user_prompt_path", '')
        if not system_prompt_path:
            raise ValueError('Empty system prompt path for agent!')
        if not user_prompt_path:
            raise ValueError('Empty user prompt path for agent!')
        self.system_prompt = read_text(system_prompt_path)
        self.user_prompt = read_text(user_prompt_path)

    def get_userprompt(self, task:str) -> None:
        self.load_promptTemp()
        self.user_prompt = self.user_prompt.replace('#task_description', task)
        if self.example_str:
            self.user_prompt = self.user_prompt.replace('#examples', self.example_str)

    def get_examplestr(self, example_paths:list = None) -> None:
        example_paths = self.params.get('example_paths', example_paths)
        if example_paths:
            examples = []
            for path in example_paths:
                examples.append(read_text(path))
            self.example_str = '\n\n'.join(['[Start of Example {}]\n{}\n[End of Example {}]'.format(i+1, ex, i+1) for i,ex in enumerate(examples)]) + '-'*40

        
    def get_messages(self, pre_messages:list = None) -> None:
        current_messages = [
            {'role': 'system', 'content': self.system_prompt},
            {'role': 'user', 'content': self.user_prompt}
        ]
        if not pre_messages:
            self.messages = current_messages
        else:
            self.messages = pre_messages + current_messages
    
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
