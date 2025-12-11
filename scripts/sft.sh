CUDA_VISIBLE_DEVICES=2 \
swift sft \
    --model /data1/fwp/pretrained_model/qwen3-8b \
    --train_type lora \
    --dataset /data1/fwp/workspace/llmeval/adaptive/MAIE/data/plan_sft.jsonl \
    --torch_dtype bfloat16 \
    --split_dataset_ratio 0.1 \
    --num_train_epochs 3 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --learning_rate 1e-4 \
    --lora_rank 8 \
    --lora_alpha 32 \
    --target_modules all-linear \
    --gradient_accumulation_steps 16 \
    --eval_steps 100 \
    --save_steps 100 \
    --save_total_limit 1 \
    --logging_steps 5 \
    --output_dir output \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 4 \
    --template qwen3
    # --max_length 2048 \
    # --model_author fwp \
    # --model_name planner-qwen3