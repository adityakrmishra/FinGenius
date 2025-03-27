#!/bin/bash
# scripts/run_pipeline.sh

# Run data processing pipeline
set -e

source venv/bin/activate

echo "🏭 Starting data pipeline..."

# Configuration
DATA_DIR="data/raw"
LOG_DIR="logs"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$LOG_DIR/pipeline_$TIMESTAMP.log"

# Create directories
mkdir -p {$DATA_DIR,$LOG_DIR}

# Run with optional test mode
if [ "$1" == "--test" ]; then
    echo "🧪 Running in test mode..."
    TEST_FLAG="--test"
    export SAMPLE_SIZE=100
fi

python -m src.data_pipeline.data_ingestor $TEST_FLAG | tee -a $LOG_FILE
python -m src.data_pipeline.data_preprocessor $TEST_FLAG | tee -a $LOG_FILE

echo "📊 Pipeline completed! Logs saved to $LOG_FILE"
