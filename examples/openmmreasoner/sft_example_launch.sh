set -x
export HF_HUB_OFFLINE=1

export NCCL_BLOCKING_WAIT=0
export TOKENIZERS_PARALLELISM=false

CONFIG=$1

# --num_processes="${ARNOLD_WORKER_GPU}" \
export FORCE_QWENVL_VIDEO_READER="decord"

#!/bin/bash

# Default values

# Put the following content into a yaml file and save it as sft_data_config.yaml
# datasets:
#     - path: /path/to/your/dataset/llava_cot.parquet
#     data_folder: "/path/to/your/dataset/images"
#     data_type: parquet
#     - path: /path/to/your/dataset/m1_sft.parquet
#     data_folder: "/path/to/your/dataset/images"
#     data_type: parquet
#     - path: /path/to/your/dataset/mmr1_filtered.parquet
#     data_folder: "/path/to/your/dataset/images"
#     data_type: parquet
#     - path: /path/to/your/dataset/OpenVLThinker-sft-iter3.parquet
#     data_folder: "/path/to/your/dataset/images"
#     data_type: parquet
#     - path: /path/to/your/dataset/WeMath.parquet
#     data_folder: "/path/to/your/dataset/images"
#     data_type: parquet
# 

DATASET_PATH="/path/to/your/dataset/sft_data_config.yaml"
PROCESSOR_NAME="Qwen/Qwen2.5-VL-7B-Instruct"
MODEL_PATH="Qwen/Qwen2.5-VL-7B-Instruct"
ATTN_IMPLEMENTATION="flash_attention_2"
PER_DEVICE_TRAIN_BATCH_SIZE=1
LEARNING_RATE=5.0e-05
WEIGHT_DECAY=0.0
GRADIENT_ACCUMULATION_STEPS=1
GRADIENT_CHECKPOINTING=true
NUM_TRAIN_EPOCHS=1
RUN_NAME="your-experiment-name"
OUTPUT_DIR="./output/your-experiment-name"
WARMUP_RATIO=0.1
MAX_STEPS=4300

torchrun --nproc_per_node $MLP_WORKER_GPU \
    --master_addr $MLP_WORKER_0_HOST \
    --node_rank $MLP_ROLE_INDEX \
    --master_port $MLP_WORKER_0_PORT \
    --nnodes $MLP_WORKER_NUM \
    -m lmms_engine.launch.cli \
    trainer_type=fsdp2_trainer \
    dataset_config.dataset_path=${DATASET_PATH} \
    dataset_config.dataset_format=yaml \
    dataset_config.processor_config.processor_name=${PROCESSOR_NAME} \
    dataset_config.dataset_type=vision_iterable \
    dataset_config.processor_config.processor_type=qwen2_5_vl \
    dataset_config.processor_config.processor_name=${PROCESSOR_NAME} \
    dataset_config.packing=true \
    dataset_config.packing_strategy=first_fit \
    dataset_config.packing_length=61440 \
    dataset_config.filter_overlong=true \
    dataset_config.video_backend=qwen_vl_utils \
    dataset_config.video_sampling_strategy=fps \
    +dataset_config.extra_kwargs.image_max_pixels=4194304 \
    model_config.load_from_pretrained_path=${MODEL_PATH} \
    model_config.attn_implementation=${ATTN_IMPLEMENTATION} \
    trainer_args.per_device_train_batch_size=${PER_DEVICE_TRAIN_BATCH_SIZE} \
    trainer_args.learning_rate=${LEARNING_RATE} \
    trainer_args.weight_decay=${WEIGHT_DECAY} \
    trainer_args.gradient_accumulation_steps=${GRADIENT_ACCUMULATION_STEPS} \
    trainer_args.gradient_checkpointing=${GRADIENT_CHECKPOINTING} \
    trainer_args.num_train_epochs=${NUM_TRAIN_EPOCHS} \
    trainer_args.warmup_ratio=${WARMUP_RATIO} \
    trainer_args.run_name=${RUN_NAME} \
    trainer_args.output_dir=${OUTPUT_DIR} \
    trainer_args.fsdp2=true \
    trainer_args.max_steps=${MAX_STEPS} \
    trainer_args.report_to=wandb \
    trainer_args.fsdp_config.transformer_layer_cls_to_wrap=["Qwen2_5_VLDecoderLayer"] \
    trainer_args.fsdp_config.reshard_after_forward=false \
    trainer_args.sp_ulysses_degree=1 \
    trainer_args.use_liger_kernel=true \
    trainer_args.use_rmpad=true \
    trainer_args.dataloader_num_workers=4 \
    trainer_args.dataloader_prefetch_factor=4 \
    trainer_args.bf16=true \
    trainer_args.lr_scheduler_type=cosine \
    trainer_args.logging_steps=1 \
    trainer_args.group_by_length=false \
    trainer_args.bf16=true