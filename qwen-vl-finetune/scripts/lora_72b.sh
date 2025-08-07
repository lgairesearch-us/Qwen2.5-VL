#!/bin/bash

#SBATCH --job-name=qwen-vl-lora
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --gres=gpu:8
#SBATCH --mem=500G
#SBATCH --time=48:00:00
#SBATCH --partition=all
#SBATCH --output=qwen_lora_%j.out
#SBATCH --error=qwen_lora_%j.err
#SBATCH --qos=low

# Create logs directory if it doesn't exist
mkdir -p logs

# Use SLURM environment for GPU count
NPROC_PER_NODE=${SLURM_GPUS_PER_NODE:-$(nvidia-smi --list-gpus | wc -l)}

# Distributed training configuration for single node
MASTER_ADDR="127.0.0.1"
MASTER_PORT=$(shuf -i 20001-29999 -n 1)
NNODES=1

# DeepSpeed configuration
deepspeed=qwen-vl-finetune/scripts/zero3.json

# Model configuration
llm=Qwen/Qwen2.5-VL-72B-Instruct  # Using HuggingFace model ID

# Training hyperparameters
lr=1e-4
batch_size=1
grad_accum_steps=4

# Training entry point
entry_file=qwen-vl-finetune/qwenvl/train/train_qwen_lora.py

# Dataset configuration (replace with public dataset names)
datasets=planning

# Output configuration
run_name=qwenvl72b_lora_bs64_lr1e-4_planning
output_dir=qwenvl72b_lora_bs64_lr1e-4_planning

# Activate conda environment
conda activate qwenvl

# Training arguments
args="
    --deepspeed ${deepspeed} \
    --model_name_or_path "${llm}" \
    --dataset_use ${datasets} \
    --data_flatten True \
    --tune_mm_vision False \
    --tune_mm_mlp True \
    --tune_mm_llm True \
    --bf16 \
    --output_dir ${output_dir} \
    --num_train_epochs 1.0 \
    --per_device_train_batch_size ${batch_size} \
    --per_device_eval_batch_size $((batch_size*2)) \
    --gradient_accumulation_steps ${grad_accum_steps} \
    --max_pixels 2330000 \
    --min_pixels 784 \
    --eval_strategy "no" \
    --save_strategy "steps" \
    --save_steps 100 \
    --save_total_limit 10 \
    --learning_rate ${lr} \
    --weight_decay 0 \
    --warmup_ratio 0.03 \
    --max_grad_norm 1 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --model_max_length 4500 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --run_name ${run_name}"

# Launch training
torchrun --nproc_per_node=${NPROC_PER_NODE} \
         --master_addr=${MASTER_ADDR} \
         --master_port=${MASTER_PORT} \
         ${entry_file} ${args}
