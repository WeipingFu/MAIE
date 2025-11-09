from utils import load_json, read_text
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from api_request import completion
from model_scheduler import get_model_key, device, GLOBAL_SCHEDULER

# GLOBAL_SCHEDULER = None

class Agent:
    def __init__(self, arg_path:str):
        self.params = load_json(arg_path)
        self.model_name = self.params.get("model_name", '')
        self.model_type = self.params.get("model_type", 'open')
        self.tokenizer_path = self.params.get('tokenizer_path','')
        self.model_path = self.params.get('model_path','')
        self.model_key = get_model_key(self.params)
        self.model = None
        self.tokenizer = None
        self.example_str = ''


    def _ensure_model_loaded(self):
        """Internal helper to ensure the necessary model is active on GPU via the scheduler."""
        # global GLOBAL_SCHEDULER
        # print('_ensure_model_loaded', GLOBAL_SCHEDULER)
        if self.model_type == 'open' and GLOBAL_SCHEDULER:
            if GLOBAL_SCHEDULER.active_model_key != self.model_key:
                # Request the scheduler to load this model and unload others
                self.model, self.tokenizer = GLOBAL_SCHEDULER.load_model(
                    self.model_key, 
                    self.model_path, 
                    self.tokenizer_path
                )
            else:
                # Model is already active, just retrieve the handles
                self.model, self.tokenizer = GLOBAL_SCHEDULER.get_active_model_and_tokenizer()
        elif self.model_type == 'open' and not GLOBAL_SCHEDULER:
             raise RuntimeError("ModelScheduler is not initialized. Cannot run open-source model agents.")
        # If model_type is not 'open' (e.g., 'api'), no model needs loading.
         
    def load_promptTemp(self):
        self.system_prompt = ''
        self.user_prompt = ''
        system_prompt_path = self.params.get("system_prompt_path", '')
        user_prompt_path = self.params.get("user_prompt_path", '')
        if system_prompt_path:
            self.system_prompt = read_text(system_prompt_path)
        if not user_prompt_path:
            raise ValueError('Empty user prompt path for agent!')
        else:
            self.user_prompt = read_text(user_prompt_path)  

    def get_userprompt(self, task:str) -> None:
        # self.load_promptTemp()
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
        current_messages = []
        if self.system_prompt:
            current_messages.append({'role': 'system', 'content': self.system_prompt})
        if self.user_prompt:
            current_messages.append({'role': 'user', 'content': self.user_prompt})
        if not pre_messages:
            self.messages = current_messages
        else:
            self.messages = pre_messages + current_messages
    
    
    def get_response(self, max_new_tokens:int=512, thinking:bool=False, prt:bool=False) -> str:
        resp = ''
        max_new_tokens = self.params.get('max_new_tokens', max_new_tokens)
        thinking = self.params.get('thinking', thinking)
        prt = self.params.get("prt", prt)
        # print('*'*50, 'Get Response', '*'*50)
        self._ensure_model_loaded()
        
        if prt:
            print('Messages:')
            print(self.messages)

        if self.model is None:
            resp = completion(self.model_name, self.messages, prt=prt)
        else:
            if self.tokenizer is None:
                raise RuntimeError("Model is loaded but tokenizer is missing.")
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
