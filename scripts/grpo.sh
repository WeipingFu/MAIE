# start planner vllm 
CUDA_VISIBLE_DEVICES=0 \
swift rollout \
    --model Qwen3-8b \
    --adapters planner-sft \
    --infer_backend 'vllm' \
    --vllm_enable_lora true \
    --host 127.0.0.1 \
    --port 8000 \
    --vllm_gpu_memory_utilization 0.9 \
    --vllm_max_lora_rank 8 \
    --temperature 1.0 \
    # --top_p 0.8




# grpo
CUDA_VISIBLE_DEVICES=1 \
swift rlhf \
    --rlhf_type grpo \
    --model Qwen3-8b \
    --adapters planner-sft \
    --ref_adapters planner-sft \
    --train_type lora \
    --external_plugins ../planner_reward.py \
    --reward_funcs planner_reward_function format \
    --use_vllm true \
    --vllm_mode server \
    --vllm_server_host 127.0.0.1 \
    --vllm_server_port 8000 \
    --dataset 'zouxuhong/Countdown-Tasks-3to4#50000' \
    --load_from_cache_file true \
    --torch_dtype bfloat16 \
    --lora_rank 8 \
    --lora_alpha 32 \
    --target_modules all-linear \
    --max_completion_length 2048 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --learning_rate 1e-6 \
    --gradient_accumulation_steps 16 \
    --eval_steps 100 \
    --save_steps 100 \
    --save_total_limit 1 \
    --logging_steps 5 \
    --output_dir output/GRPO_COUNTDOWN \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 4 \
    --num_generations 8 \
    --temperature 0.7 \
    --top_p 0.8 \
    --deepspeed zero3 \
    --log_completions true \
    --report_to wandb \
    --beta 0.0 \
    --num_iterations 1