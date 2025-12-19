CUDA_VISIBLE_DEVICES=0 \
swift sft \
    --model /autodl-fs/data/pretrained_model/qwen3-8b \
    --train_type lora \
    --dataset /autodl-fs/data/maie/data/plan_sft_train.jsonl \
    --val_dataset /autodl-fs/data/maie/data/plan_sft_eval.jsonl \
    --torch_dtype bfloat16 \
    --num_train_epochs 3 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --learning_rate 5e-5 \
    --lora_rank 8 \
    --lora_alpha 32 \
    --target_modules all-linear \
    --gradient_accumulation_steps 16 \
    --eval_steps 100 \
    --save_steps 100 \
    --save_total_limit 1 \
    --early_stop_interval 3 \
    --logging_steps 5 \
    --output_dir /autodl-fs/data/maie/model/planner \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 4 \
    --template qwen3
    # --max_length 2048 \
    # --model_author fwp \
    # --model_name planner-qwen3