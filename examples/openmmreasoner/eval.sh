cd /path/to/OpenMMReasoner

export VLLM_WORKER_MULTIPROC_METHOD=spawn
export OPENAI_API_KEY="EMPTY"
export OPENAI_BASE_URL="http://llm-judge-address:8000/v1"
export OPENAI_MODEL_NAME="judge"
export USE_LLM_JUDGE="True"
export LMMS_EVAL_USE_CACHE=True
export LMMS_EVAL_HOME=/path/to/.cache/lmms-eval

CKPT_PATH=$1
TASK_NAME=$2

accelerate launch --num_processes=8 --main_process_port=12346 -m lmms_eval \
    --model vllm \
    --model_args model=$CKPT_PATH,tensor_parallel_size=1,data_parallel_size=8,disable_log_stats=True,gpu_memory_utilization=0.7 \
    --tasks $TASK_NAME \
    --batch_size 256 \
    --log_samples \
    --output_path ./logs --verbosity DEBUG  --include_path ./lmms_eval_tasks
