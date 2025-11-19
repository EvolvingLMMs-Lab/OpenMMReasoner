#!/bin/bash

# Download OpenMMReasoner datasets using Hugging Face CLI
# This script downloads:
#   - OpenMMReasoner/OpenMMReasoner-SFT-874K
#   - OpenMMReasoner/OpenMMReasoner-RL-74K

# Set default local directory
LOCAL_DIR="${1:-.}/data"

# Create the local directory if it doesn't exist
mkdir -p "$LOCAL_DIR"

echo "Downloading OpenMMReasoner datasets to $LOCAL_DIR"

# Download OpenMMReasoner-SFT-874K
echo "Downloading OpenMMReasoner-SFT-874K..."
huggingface-hub download \
    --repo-type dataset \
    --local-dir "$LOCAL_DIR/OpenMMReasoner-SFT-874K" \
    OpenMMReasoner/OpenMMReasoner-SFT-874K

if [ $? -ne 0 ]; then
    echo "Error downloading OpenMMReasoner-SFT-874K"
    exit 1
fi

echo "Successfully downloaded OpenMMReasoner-SFT-874K"

# Download OpenMMReasoner-RL-74K
echo "Downloading OpenMMReasoner-RL-74K..."
huggingface-hub download \
    --repo-type dataset \
    --local-dir "$LOCAL_DIR/OpenMMReasoner-RL-74K" \
    OpenMMReasoner/OpenMMReasoner-RL-74K

if [ $? -ne 0 ]; then
    echo "Error downloading OpenMMReasoner-RL-74K"
    exit 1
fi

echo "Successfully downloaded OpenMMReasoner-RL-74K"
echo "All datasets downloaded successfully to $LOCAL_DIR"

