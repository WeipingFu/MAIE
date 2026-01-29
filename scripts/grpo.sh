# export CUDA_HOME=/usr/local/cuda
# export PATH=$CUDA_HOME/bin:$PATH
# export CPATH=$CUDA_HOME/include
# export LIBRARY_PATH=$CUDA_HOME/lib64
# export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

which nvcc
nvcc --version
export PYTHONPATH=$PYTHONPATH:$(pwd)

# grpo
CUDA_VISIBLE_DEVICES=0,1 \
swift rlhf \
    --rlhf_type grpo \
    --model /root/autodl-tmp/pretrained_model/qwen3-8b \
    --adapters /root/autodl-tmp/maie/model/planner/checkpoint-87 \
    --ref_adapters /root/autodl-tmp/maie/model/planner/checkpoint-87 \
    --train_type lora \
    --external_plugins /root/autodl-tmp/maie/code/planner_reward.py \
    --reward_funcs planner_reward_function \
    --use_vllm true \
    --vllm_mode colocate \
    --dataset /root/autodl-tmp/maie/data/grpo_r1.jsonl \
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
    --output_dir /root/autodl-tmp/maie/model/planner/grpo \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 4 \
    --num_generations 8 \
    --temperature 1.0 \
    --log_completions true \
    --beta 0.0 \
    --num_iterations 1 \
    --deepspeed zero3 \
    --sleep_level 1 \
    --offload_optimizer true \
    --offload_model true \
    --vllm_gpu_memory_utilization 0.5 \
    --vllm_max_lora_rank 8 \
    --vllm_enable_lora true \
    
    # --vllm_server_host 127.0.0.1 \
    # --vllm_server_port 8000 \