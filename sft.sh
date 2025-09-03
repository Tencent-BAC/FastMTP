# Configurations
per_device_batch_size=1
nproc_per_node=8
gradient_accumulation_step=8
max_length_token=4096

# Model and Data Paths (Please modify according to your setup)
model_path=/absolute/path/to/model/MiMo-7B-RL
dataset_path=/absolute/path/to/data/dataset.jsonl
output_path=/absolute/path/to/output

# Training Script
PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
NPROC_PER_NODE=$nproc_per_node \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
swift sft \
    --model $model_path \
    --model_type 'mimo'\
    --train_type full \
    --num_train_epochs 3 \
    --dataset $dataset_path \
    --freeze_parameters model,lm_head \
    --trainable_parameters model.mtp_layers \
    --torch_dtype bfloat16 \
    --per_device_train_batch_size $per_device_batch_size \
    --per_device_eval_batch_size $per_device_batch_size \
    --learning_rate 5e-5 \
    --gradient_accumulation_steps $gradient_accumulation_step \
    --eval_steps 1000 \
    --save_steps 1000 \
    --save_total_limit 2 \
    --logging_steps 5 \
    --max_length $max_length_token \
    --truncation_strategy 'delete' \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 8 \
    --dataset_num_proc 8 \
    --output_dir $output_path \
    --attn_impl flash_attn
    