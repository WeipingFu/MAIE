# start planner vllm 
CUDA_VISIBLE_DEVICES=0 \
swift rollout \
    --model /root/autodl-tmp/pretrained_model/qwen3-8b \
    --adapters /root/autodl-tmp/maie/model/planner/checkpoint-87 \
    --infer_backend 'vllm' \
    --vllm_enable_lora true \
    --host 0.0.0.0 \
    --port 8000 \
    --vllm_gpu_memory_utilization 0.5 \
    --vllm_max_lora_rank 8 \
    --temperature 1.0 \
    # --top_p 0.8
