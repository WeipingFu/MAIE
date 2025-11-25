import asyncio
import pandas as pd
from tqdm import tqdm
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

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
    input_path = "/data/fwp/workspace/benchmarks/pairwise/mt-bench/human.jsonl"
    output_path = "/data/fwp/workspace/adaptive/result/mt-bench/plan-judge-llama3.xlsx"

    mtbench = load_jsonl(input_path)
    # mtbench = mtbench[0:1]
    print(f'------------------------Start testing. Load {len(mtbench)} data.------------------------')
    new_data = []
    fail_count = 0
    for idx, item in enumerate(mtbench):
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
        evaluation_plan, judge_results, final_judgement = await run_pipeline(
            task=task,
            model_responses=model_responses,
            convs=convs,
            with_critic=False,
            judge_chat=False
        )
        item['evaluation_plan'] = evaluation_plan
        item['judge_results'] = judge_results
        item['judgement'] = final_judgement
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
