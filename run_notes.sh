# train single gpu
CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --nnodes=1 train_sft_lora.py

# accelerate train multi gpu
export CUDA_VISIBLE_DEVICES=0,1
accelerate config
accelerate launch train_sft_lora.py