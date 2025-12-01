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
    input_path = "/autodl-fs/data/maie/benchmarks/pairwise/mt-bench/human.jsonl"
    output_path = "/autodl-fs/data/maie/result/mt-bench/plan-critic-3-judge-true-qwen3-8b-1.xlsx"

    mtbench = load_jsonl(input_path)
    mtbench = mtbench[11:]
    # mtbench = pd.read_excel(input_path)
    # mtbench = mtbench[mtbench['judgement'].isna()].to_dict(orient='records')
    
    print(f'------------------------Start testing. Load {len(mtbench)} data.------------------------')
    new_data = []
    fail_count = 0
    for idx, item in tqdm(enumerate(mtbench), total=len(mtbench)):
        print(f'Handle the item idx={idx}')
        convs = None
        if int(item['turn']) > 1:
            conv_a = item['conversation_a'][:-2]
            conv_b = item['conversation_b'][:-2]
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
            critic_round=3,
            judge_chat=True
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
    results = asyncio.run(test_mtbench())
