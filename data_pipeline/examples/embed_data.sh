
export HF_HUB_OFFLINE=1

CONFIG_PATH=$1
CACHE_FOLDER=${2:-cache/embed}
SOURCE_TYPE=${3:-rl}

torchrun --nproc_per_node=8 src/openmmdata/preprocess/embed.py \
    --config $CONFIG_PATH \
    --cache-folder $CACHE_FOLDER \
    --source-type $SOURCE_TYPE