deepspeed  \
    finetune_hgface.py \
    --output_dir yotta-test-128-bigdata2-loss2.0-finetune1.0 \
    --num_train_epochs 1 \
    --dataloader_drop_last True \
    --warmup_ratio 0.30 \
    --per_device_eval_batch_size 512 \
    --per_device_train_batch_size 512 \
    --temperature 0.01 \
    --logging_steps 1 \
    --run_name yotta-test-128-bigdata2-loss2.0-finetune1.0 \
    --save_total_limit 20 \
    --weight_decay 0.0001 \
    --learning_rate 0.000003 \
    --save_steps 50 \
    --deepspeed "dp_config_finetune.json" \
    --report_to wandb \
    --gradient_checkpointing
