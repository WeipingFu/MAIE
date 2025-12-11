import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import pandas as pd
import numpy as np
from .utils import load_jsonl, save_jsonl, clean_json
from .agents.PlanAgent import UserPrompt, PlannerResponse
from swift.llm import (
    PtEngine, RequestConfig, safe_snapshot_download, get_model_tokenizer, get_template, InferRequest
)
from swift.tuners import Swift


def plan_infer(model, tokenizer, template_type, data_path, save_path):
    data = load_jsonl(data_path)
    user_prompt = UserPrompt(mode='plan')
    
    default_system = 'You are the Planning Agent in a multi-agent system. Your function is to design a detailed, structured evaluation plan for the given instance input. If the input contains revision feedback, revise the existing plan accordingly.'
    template_type = template_type or model.model_meta.template
    template = get_template(template_type, tokenizer, default_system=default_system)
    engine = PtEngine.from_model_template(model, template, max_batch_size=8)
    request_config = RequestConfig(max_tokens=2048, temperature=0)

    infer_requests = []
    for idx, item in enumerate(data):
        convs = item['conversations']
        if isinstance(convs, np.ndarray):
            convs = convs.tolist()
        content = user_prompt.generate_user_prompt(
            task=item['question'], 
            model_responses=item['model_response'], 
            eval_mode=item['eval_type'],
            original_evaluation_plan=None,
            feedback=None, 
            convs=convs,
            criteria_list=None
        )
        infer_requests.append(InferRequest(messages=[{'role': 'user', 'content': content}]))
    
    resp_list = engine.infer(infer_requests, request_config)

    results = []
    for idx, resp in enumerate(resp_list):
        query = infer_requests[idx].messages[0]['content']
        eval_plan = resp.choices[0].message.content
        eval_plan = clean_json(eval_plan)
        try:
            parsed = PlannerResponse.model_validate_json(eval_plan)
            clean_json_str = parsed.model_dump_json(indent=2)
        except Exception as e:
            print("Warning: PlannerResponse validation failed. Exception:", e)
            clean_json_str = ''

        if not clean_json_str:
            one = {k:v for k,v in data[idx].items()}
            one['query'] = query
            one['evaluation_plan'] = clean_json_str
            results.append(one)
    
    save_jsonl(results, save_path)



if __name__ == "__main__":
    model = '' 
    template_type = 'qwen3' 
    model, tokenizer = get_model_tokenizer(model)
    lora_checkpoint = safe_snapshot_download('') 
    if lora_checkpoint is not None:
        model = Swift.from_pretrained(model, lora_checkpoint)

    data_path = ''
    save_path = ''
    plan_infer(model, tokenizer, template_type, data_path, save_path)