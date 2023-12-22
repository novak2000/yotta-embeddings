deepspeed  \
    train_hgface.py \
    --output_dir yotta-test-128-bigdata2 \
    --num_train_epochs 2 \
    --dataloader_drop_last True \
    --warmup_ratio 0.10 \
    --per_device_eval_batch_size 4096 \
    --per_device_train_batch_size 4096 \
    --temperature 0.01 \
    --logging_steps 1 \
    --run_name yotta-test-128-bigdata2 \
    --save_total_limit 20 \
    --weight_decay 0.0001 \
    --learning_rate 0.0003 \
    --save_steps 200 \
    --deepspeed "dp_config.json" \
    --fp16 \
    --report_to wandb \
    --gradient_checkpointing
