
export HF_HUB_OFFLINE=1

CONFIG_PATH=$1
CACHE_FOLDER=${2:-cache/embed}
SOURCE_TYPE=${3:-rl}
DATASET_CACHE_FOLDER=${4:-cache/deduplicate}


python src/openmmdata/preprocess/deduplicate.py \
    --config $CONFIG_PATH \
    --cache-folder $CACHE_FOLDER \
    --dataset-cache-folder $DATASET_CACHE_FOLDER \
    --source-type $SOURCE_TYPE \
    --threshold 0.99 \
    --nprocess 8 # Number of torch devices that you use