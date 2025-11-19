
# Distill model

# Serve Config
export MODEL_VERSION=qwen3-vl
export OAI_BASE_URL="http://xxx:8000/v1" # Change to your own server address
export OAI_API_KEY="EMPTY"


# LLM AS JUDGE
export USE_LLM_JUDGE="True"
export OPENAI_BASE_URL="http://xxx:8000/v1" # Change to your own judge server address
export OPENAI_API_KEY='EMPTY'
export OPENAI_MODEL_NAME='judge'


python distill_response.py \
  --config /path/to/example_config.yaml \
  --output-folder /path/to/output_folder \
  --num_rollouts 2 # Change to your own number of rollouts