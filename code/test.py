import asyncio
import pandas as pd
from tqdm import tqdm
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from .main import run_pipeline          
from .utils import load_jsonl, save_jsonl  

def save_results(data, output_path):
    if output_path.endswith(".xlsx"):
        pd.DataFrame(data).to_excel(output_path, index=False)
        print(f"Saved {len(data)} results to {output_path}")
    elif output_path.endswith(".jsonl"):
        save_jsonl(data, output_path)
        print(f"Saved {len(data)} results to {output_path}")
    else:
        print("Output format is not supported! Results are not saved!")


async def test_mtbench():
    input_path = "/data1/fwp/workspace/llmeval/adaptive/result/mt-bench/plan-critic-judge-chat-qwen3-8b.xlsx"
    output_path = "/data1/fwp/workspace/llmeval/adaptive/result/mt-bench/plan-critic-judge-chat-qwen3-8b-2.xlsx"

    # mtbench = load_jsonl(input_path)
    # mtbench = mtbench[0:]
    mtbench = pd.read_excel(input_path)
    mtbench = mtbench[mtbench['judgement'].isna()].to_dict(orient='records')
    
    print(f'------------------------Start testing. Load {len(mtbench)} data.------------------------')
    new_data = []
    fail_count = 0
    for idx, item in tqdm(enumerate(mtbench), total=len(mtbench)):
        print(f'Handle the item idx={idx}')
        convs = None
        if int(item['turn']) > 1:
            conv_a = eval(item['conversation_a'])[:-2]
            conv_b = eval(item['conversation_b'])[:-2]
            convs = []
            for i in range(0, len(conv_a), 2):
                convs.append({
                    'user': conv_a[i]['content'],
                    'a': conv_a[i+1]['content'],
                    'b': conv_b[i+1]['content']
                })
        # ==== run_pipeline ====
        task = item['question']
        model_responses = [item['model_a_response'], item['model_b_response']]
        evaluation_plan, judge_results, final_judgement, plan_revise_count, judge_revise_count = await run_pipeline(
            task, 
            model_responses, 
            eval_mode='pairwise',
            convs=convs,
            criteria_list=None,
            critic_round=3,
            judge_chat=True,
            allow_tie=True
        )
        item['evaluation_plan'] = evaluation_plan
        item['judge_results'] = judge_results
        item['judgement'] = final_judgement
        item['plan_revise_count'] = plan_revise_count
        item['judge_revise_count'] = judge_revise_count

        if not final_judgement:
            fail_count += 1

        new_data.append(item)
        if len(new_data) % 2 and len(new_data) > 0:
            save_results(new_data, output_path)
            
    # --- save results ---
    save_results(new_data, output_path)
    print(f'------------------------All test data done! Fail count: {fail_count}------------------------')
    return new_data


async def test_judgebench():
    input_path = "/data1/fwp/workspace/llmeval/adaptive/result/judgebench/plan-judge-chat-qwen3-8b.jsonl"
    output_path = "/data1/fwp/workspace/llmeval/adaptive/result/judgebench/qwen3-8b-null.jsonl"

    bench = load_jsonl(input_path)
    bench = bench[0:]
    bench = [x for x in bench if not x['judgement']]
    # mtbench = pd.read_excel(input_path)
    # mtbench = mtbench[mtbench['judgement'].isna()].to_dict(orient='records')
    
    print(f'------------------------Start testing. Load {len(bench)} data.------------------------')
    new_data = []
    fail_count = 0
    for idx, item in tqdm(enumerate(bench), total=len(bench)):
        print(f'Handle the item idx={idx}')
        convs = None
        # ==== run_pipeline ====
        task = item['question']
        model_responses = [item['response_A'], item['response_B']]
        winner = ''
        if item['label'] == 'A>B':
            winner = 'model_a'
        elif item['label'] == 'B>A':
            winner = 'model_b'
        evaluation_plan, judge_results, final_judgement, plan_revise_count, judge_revise_count = await run_pipeline(
            task, 
            model_responses, 
            eval_mode='pairwise',
            convs=convs,
            criteria_list=None,
            critic_round=0,
            judge_chat=True,
            allow_tie=False
        )
        item['winner'] = winner
        item['evaluation_plan'] = evaluation_plan
        item['judge_results'] = judge_results
        item['judgement'] = final_judgement
        item['plan_revise_count'] = plan_revise_count
        item['judge_revise_count'] = judge_revise_count

        if not final_judgement:
            fail_count += 1

        new_data.append(item)
        if len(new_data) % 2 and len(new_data) > 0:
            save_results(new_data, output_path)
            
    # --- save results ---
    save_results(new_data, output_path)
    print(f'------------------------All test data done! Fail count: {fail_count}------------------------')
    return new_data


async def test_feedbackbench(enable_criteria=True):
    tail = '-criteria' if enable_criteria else ''
    input_path = "/data1/fwp/workspace/llmeval/benchmarks/pointwise/feedbackbench/feedbackbench_test.jsonl"
    output_path = f"/data1/fwp/workspace/llmeval/adaptive/result/feedbackbench/plan-critic-judge-chat-qwen3-8b{tail}.jsonl"

    bench = load_jsonl(input_path)
    bench = bench[0:]
    # mtbench = pd.read_excel(input_path)
    # mtbench = mtbench[mtbench['judgement'].isna()].to_dict(orient='records')
    
    print(f'------------------------Start testing. Load {len(bench)} data.------------------------')
    new_data = []
    fail_count = 0
    for idx, item in tqdm(enumerate(bench), total=len(bench)):
        print(f'Handle the item idx={idx}')
        convs = None
        # ==== run_pipeline ====
        task = item['task_description']
        model_responses = [item['model_response']]
        criteria_list = None
        if enable_criteria:
            criteria_list = [{
                "dimension": item['dimension'], 
                "description": item['dimension_description'], 
                "scoring_scale":"1-5", 
                "high_score_indicator": '\n'.join(item['score_indicators'][3:]), 
                "low_score_indicator": '\n'.join(item['score_indicators'][0:3])
            }]
        evaluation_plan, judge_results, final_judgement, plan_revise_count, judge_revise_count = await run_pipeline(
            task, 
            model_responses, 
            eval_mode='pointwise',
            convs=convs,
            criteria_list=criteria_list,
            critic_round=3,
            judge_chat=True,
            target_min=1.0,
            target_max=5.0
        )

        item['evaluation_plan'] = evaluation_plan
        item['judge_results'] = judge_results
        item['judgement'] = final_judgement
        item['plan_revise_count'] = plan_revise_count
        item['judge_revise_count'] = judge_revise_count

        if not final_judgement:
            fail_count += 1

        new_data.append(item)
        if len(new_data) % 2 and len(new_data) > 0:
            save_results(new_data, output_path)
            
    # --- save results ---
    save_results(new_data, output_path)
    print(f'------------------------All test data done! Fail count: {fail_count}------------------------')
    return new_data


async def test_rewardbench():
    input_path = '/data1/fwp/workspace/llmeval/adaptive/result/rewaredbench/plan-critic-judge-chat-qwen3-8b.jsonl'
    output_path = f'/data1/fwp/workspace/llmeval/adaptive/result/rewaredbench/plan-critic-judge-chat-qwen3-8b-1.jsonl'

    bench = load_jsonl(input_path)
    bench = bench[0:]
    bench = [x for x in bench if not x['judgement'] or x['judgement']=='tie']
    # mtbench = pd.read_excel(input_path)
    # mtbench = mtbench[mtbench['judgement'].isna()].to_dict(orient='records')
    
    print(f'------------------------Start testing. Load {len(bench)} data.------------------------')
    new_data = []
    fail_count = 0
    for idx, item in tqdm(enumerate(bench), total=len(bench)):
        print(f'Handle the item idx={idx}')
        # ==== run_pipeline ====
        task = item['question']
        model_responses = [item['response_a'], item['response_b']]
       
        evaluation_plan, judge_results, final_judgement, plan_revise_count, judge_revise_count = await run_pipeline(
            task, 
            model_responses, 
            eval_mode='pairwise',
            convs=None,
            criteria_list=None,
            critic_round=3,
            judge_chat=True,
            allow_tie=False
        )
        item['evaluation_plan'] = evaluation_plan
        item['judge_results'] = judge_results
        item['judgement'] = final_judgement
        item['plan_revise_count'] = plan_revise_count
        item['judge_revise_count'] = judge_revise_count

        if not final_judgement:
            fail_count += 1

        new_data.append(item)
        if len(new_data) % 2 and len(new_data) > 0:
            save_results(new_data, output_path)
            
    # --- save results ---
    save_results(new_data, output_path)
    print(f'------------------------All test data done! Fail count: {fail_count}------------------------')
    return new_data


async def test_rewardbench2():
    input_path = '/data1/fwp/workspace/llmeval/adaptive/result/rewardbench2/plan-critic-judge-chat-qwen3-8b.jsonl'
    output_path = f'/data1/fwp/workspace/llmeval/adaptive/result/rewardbench2/plan-critic-judge-chat-qwen3-8b-1.jsonl'

    bench = load_jsonl(input_path)
    bench = bench[0:]
    bench = [x for x in bench if not x['judgement'] or x['judgement']=='tie']
    
    print(f'------------------------Start testing. Load {len(bench)} data.------------------------')
    new_data = []
    fail_count = 0
    for idx, item in tqdm(enumerate(bench), total=len(bench)):
        print(f'Handle the item idx={idx}')
        # ==== run_pipeline ====
        task = item['question']
        model_responses = [item['response_a'], item['response_b']]
       
        evaluation_plan, judge_results, final_judgement, plan_revise_count, judge_revise_count = await run_pipeline(
            task, 
            model_responses, 
            eval_mode='pairwise',
            convs=None,
            criteria_list=None,
            critic_round=3,
            judge_chat=True,
            allow_tie=False
        )
        item['evaluation_plan'] = evaluation_plan
        item['judge_results'] = judge_results
        item['judgement'] = final_judgement
        item['plan_revise_count'] = plan_revise_count
        item['judge_revise_count'] = judge_revise_count

        if not final_judgement:
            fail_count += 1

        new_data.append(item)
        if len(new_data) % 2 and len(new_data) > 0:
            save_results(new_data, output_path)
            
    # --- save results ---
    save_results(new_data, output_path)
    print(f'------------------------All test data done! Fail count: {fail_count}------------------------')
    return new_data


async def test_flask(enable_criteria=True):
    tail = '-criteria' if enable_criteria else ''
    input_path = f"/data1/fwp/workspace/llmeval/benchmarks/pointwise/flask_eval.jsonl"
    output_path = f"/data1/fwp/workspace/llmeval/adaptive/result/flask/plan-critic-judge-chat-qwen3-8b{tail}.jsonl"

    bench = load_jsonl(input_path)
    bench = bench[0:]
    # bench = [x for x in bench if not x['judgement'] or float(x['judgement'])<1 or float(x['judgement'])>5]
    
    print(f'------------------------Start testing. Load {len(bench)} data.------------------------')
    new_data = []
    fail_count = 0
    for idx, item in tqdm(enumerate(bench), total=len(bench)):
        print(f'Handle the item idx={idx}')
        convs = None
        # ==== run_pipeline ====
        task = item['task_instruction']
        model_responses = [item['model_response']]
        criteria_list = None
        if enable_criteria:
            criteria_list = [{
                "dimension": item['dimension'], 
                "description": item['dimension_description'], 
                "scoring_scale":"1-5", 
                "high_score_indicator": '\n'.join(item['score_indicators'][3:]), 
                "low_score_indicator": '\n'.join(item['score_indicators'][0:3])
            }]
        evaluation_plan, judge_results, final_judgement, plan_revise_count, judge_revise_count = await run_pipeline(
            task, 
            model_responses, 
            eval_mode='pointwise',
            convs=convs,
            criteria_list=criteria_list,
            critic_round=3,
            judge_chat=True,
            target_min=1.0,
            target_max=5.0
        )

        item['evaluation_plan'] = evaluation_plan
        item['judge_results'] = judge_results
        item['judgement'] = final_judgement
        item['plan_revise_count'] = plan_revise_count
        item['judge_revise_count'] = judge_revise_count

        if not final_judgement:
            fail_count += 1

        new_data.append(item)
        if len(new_data) % 2 and len(new_data) > 0:
            save_results(new_data, output_path)
            
    # --- save results ---
    save_results(new_data, output_path)
    print(f'------------------------All test data done! Fail count: {fail_count}------------------------')
    return new_data


if __name__ == "__main__":
    # pairwise
    # results = asyncio.run(test_mtbench())
    results = asyncio.run(test_judgebench())
    # results = asyncio.run(test_rewardbench())
    # results = asyncio.run(test_rewardbench2())

    # pointwise
    # results = asyncio.run(test_feedbackbench(enable_criteria=True))
    # results = asyncio.run(test_flask(enable_criteria=True))