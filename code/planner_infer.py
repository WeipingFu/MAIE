import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import pandas as pd
import numpy as np
from .utils import read_text, load_jsonl, save_jsonl, clean_json
# from .agents.PlanAgent import UserPrompt, PlannerResponse
from pydantic import BaseModel, Field
from typing import List, Literal, Optional
from jinja2 import Template
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
# from swift.llm import (
#     PtEngine, RequestConfig, safe_snapshot_download, get_model_tokenizer, get_template, InferRequest, BaseArguments
# )
# from swift.tuners import Swift


# --- JudgeAgent Details ---
class AgentDetails(BaseModel):
    """Defines the role, task, and specific steps for the evaluation Agent."""
    role_name: str = Field(description="The name of the JudgeAgent.")
    role_description: str = Field(description="The role the Agent plays and its primary responsibilities.")
    evaluation_task: str = Field(description="A concise description of the evaluation task the Agent must judge.")
    evaluation_steps: List[str] = Field(description="Detailed, step-by-step guidelines that the Agent must follow.")

# --- Evaluation Dimension ---
class EvaluationDimension(BaseModel):
    """Defines a single dimension within a multi-dimensional evaluation system."""
    name: str = Field(description="The name of the dimension.")
    definition: str = Field(description="A definition describing what this dimension measures.")
    rationale: str = Field(description="Explanation of why this dimension is important for the evaluation.")
    weight: float = Field(description="The weight of this dimension in the final total score (0.0-1.0). The sum of all weights should be 1.")
    evaluation_granularity: str = Field(
        description='The scope and granularity of the evaluation: "holistic" (judges the response as a whole), "localized" (focuses on specific portions), "stepwise" (examines each reasoning or generation step).'
    )
    # evaluation_granularity: Literal["holistic", "localized", "stepwise"] = Field(
    #     description='The scope and granularity of the evaluation: "holistic" (judges the response as a whole), "localized" (focuses on specific portions), "stepwise" (examines each reasoning or generation step).'
    # )
    scoring_scale: str = Field(description='The scoring system used by the evaluator, e.g., "1-5", "0-100", or "binary preference".')
    high_score_indicator: str = Field(description="Describes the characteristics of a high score (full marks).")
    low_score_indicator: str = Field(description="Describes the characteristics of a low score (minimum marks).")
    dependencies: Optional[List[str]] = Field(default_factory=list, description="Optional list of names of other dimensions this dimension related with.")
    assigned_agent: AgentDetails = Field(description="The configuration for the JuegeAgent assigned to perform this dimension's evaluation.")

# --- Format of Planner Response ---
class PlannerResponse(BaseModel):
    """
    Structured response returned by the Planner Agent, used to guide subsequent evaluation flows.
    """
    task_type: str = Field(description="Identifies the type of task, e.g., 'Code Generation', 'Summarization'.")
    sub_task: str = Field(description="The specific sub-task.")
    evaluation_mode: Literal["pointwise", "pairwise"] = Field(
        description="Specifies whether the evaluation targets a single response (pointwise) or compares two responses (pairwise)."
    )
    evaluation_dimensions: List[EvaluationDimension] = Field(
        description="A list of multi-dimensional criteria to be used for evaluating the generation result."
    )


# --- User prompt for Planner ---
class UserPrompt:
    def __init__(self, mode='plan'):
        self._mode = mode
        if self._mode == 'revise':
            self._user_prompt = read_text("./prompts/plan-revise.txt")
        else:
            self._user_prompt = read_text("./prompts/plan.txt")

    def get_conv_history(self, convs):
        history_str = ''
        if convs and len(convs) > 0:
            history_str = '[Conversation History Between User and Models]\n'
            for idx, conv in enumerate(convs):
                history_str += '[Turn {}]\n[User]\n{}\n[Model a]\n{}\n[Model b]\n{}\n\n'.format(str(idx+1), conv['user'], conv['a'], conv['b'])
        return history_str
    
    def get_user_criteria(self, eval_mode, criteria_list=None):
        user_criteria = ''
        scoring_scale = 'binary preference' if eval_mode == 'pairwise' else '1-5'
        if criteria_list and len(criteria_list) > 0:
            # [{"dimension":"", "description":"", "scoring_scale":"", "high_score_indicator":"", "low_score_indicator":""}, ...]
            user_criteria += '[User-defined Evaluation Criteria]\n'
            for i, item in enumerate(criteria_list):
                dimension_str = item.get('dimension','')+"-"+item.get('description')
                one = f"Dimension: {dimension_str}\nScoring Scale: {item.get('scoring_scale', scoring_scale)}\nHigh Score Indicator: {item.get('high_score_indicator','')}\nLow Score Indicator: {item.get('low_score_indicator','')}\n"
                user_criteria += one
        return user_criteria
    
    def get_model_response(self, model_responses):
        mapping = {i: chr(ord('a') + i - 1) for i in range(1, 27)}
        if len(model_responses) == 1:
            model_response_str = model_responses[0]
        else:
            model_response_str = '\n'.join([f'[Response of Model {mapping[idx+1]}]\n{response}' for idx, response in enumerate(model_responses)])
        return model_response_str

    def generate_user_prompt(self, task, model_responses, eval_mode=None, original_evaluation_plan=None, feedback=None, convs=None, criteria_list=None):
        if not eval_mode:
            if len(model_responses) == 1:
                eval_mode = 'pointwise'
            elif len(model_responses) == 2:
                eval_mode = 'pairwise'
            else:
                raise ValueError('The count of model_responses = {}, which is not supported!'.format(len(model_responses)))
        template_vars = {
            "examples": '',
            "history": self.get_conv_history(convs),
            "task_description": task,
            "model_response": self.get_model_response(model_responses),
            "criteria": self.get_user_criteria(eval_mode, criteria_list)    
        }
        if self._mode == 'revise':
            template_vars["original_evaluation_plan"] = original_evaluation_plan
            template_vars["feedback"] = feedback
        else:
            template_vars["evaluation_mode"] = eval_mode
        template = Template(self._user_prompt)
        content = template.render(**template_vars)
        # print(f'User Prompt:\n{content}')
        return content



def plan_infer(model, tokenizer, lora_path, data_path, save_path):
    data = load_jsonl(data_path)
    user_prompt = UserPrompt(mode='plan')
    
    # template_type = template_type or model.model_meta.template
    # template = get_template(template_type, tokenizer)
    # engine = PtEngine.from_model_template(model, template, max_batch_size=8)
    # request_config = RequestConfig(max_tokens=2048, temperature=0)

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
        # print(f'User Prompt:\n{content}')
        messages = [
            {'role': 'system', 'content': 'You are the Planning Agent in a multi-agent system. Your function is to design a detailed, structured evaluation plan for the given instance input. If the input contains revision feedback, revise the existing plan accordingly.'},
            {'role': 'user', 'content': content}
        ]
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False
        )
        infer_requests.append(prompt)    
        # infer_requests.append(InferRequest(messages=[
        #     {'role': 'system', 'content': 'You are the Planning Agent in a multi-agent system. Your function is to design a detailed, structured evaluation plan for the given instance input. If the input contains revision feedback, revise the existing plan accordingly.'},
        #     {'role': 'user', 'content': content}]
        # ))
    
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=2048
    )
    # resp_list = engine.infer(infer_requests, request_config)
    outputs = model.generate(
        infer_requests, 
        sampling_params, 
        lora_request=LoRARequest("planner_adapter", 1, lora_path),
    )
    print(len(outputs))
    results = []
    for idx, output in enumerate(outputs):
        # query = infer_requests[idx].messages[0]['content']
        # eval_plan = resp.choices[0].message.content
        eval_plan = output.outputs[0].text.strip()
        eval_plan = clean_json(eval_plan)
        # try:
        #     parsed = PlannerResponse.model_validate_json(eval_plan)
        #     clean_json_str = parsed.model_dump_json(indent=2)
        # except Exception as e:
        #     print("Warning: PlannerResponse validation failed. Exception:", e)
        #     clean_json_str = ''

        # if not clean_json_str:
        one = {k:v for k,v in data[idx].items()}
        # one['query'] = query
        one['evaluation_plan'] = eval_plan
        results.append(one)
    
    save_jsonl(results, save_path)



if __name__ == "__main__":
    model_path = '/autodl-fs/data/pretrained_model/qwen3-8b' 
    lora_path = "/autodl-fs/data/maie/model/planner-sft-qwen3-8b"
    model = LLM(
        model=model_path,
        tokenizer=model_path,
        enable_lora=True, 
        dtype="bfloat16",
        gpu_memory_utilization=0.9
    )
    tokenizer = model.get_tokenizer()
    # model, tokenizer = get_model_tokenizer(model)
    # lora_checkpoint = safe_snapshot_download('/autodl-fs/data/maie/model/planner-sft-qwen3-8b') 
    # if lora_checkpoint is not None:
    #     model = Swift.from_pretrained(model, lora_checkpoint)
    # args = BaseArguments.from_pretrained(lora_checkpoint)
    # print(f'args.model: {args.model}')
    # print(f'args.model_type: {args.model_type}')
    # print(f'args.template_type: {args.template}')
    # print(f'args.default_system: {args.system}')

    data_path = '/autodl-fs/data/maie/data/for_planner.jsonl'
    save_path = '/autodl-fs/data/maie/data/planner_result.jsonl'
    plan_infer(model, tokenizer, lora_path, data_path, save_path)