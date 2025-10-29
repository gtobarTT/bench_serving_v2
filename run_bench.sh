#!/bin/bash

# Bash script to run benchmark_serving.py for each row in benchmark.csv
# Usage: ./run_bench.sh <model_name> [base_url] [backend]
# Example: ./run_bench.sh Qwen/Qwen2.5-3B-Instruct http://localhost:8000 vllm

set -e  # Exit on error

# Check if model name is provided
if [ $# -lt 1 ]; then
    echo "Usage: $0 <model_name> [base_url] [backend]"
    echo "Example: $0 Qwen/Qwen2.5-3B-Instruct http://localhost:8000 vllm"
    exit 1
fi

MODEL="$1"
BASE_URL="${2:-http://localhost:8000}"
BACKEND="${3:-vllm}"
DEVICE="gpu"

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CSV_FILE="${SCRIPT_DIR}/benchmark.csv"

# Check if CSV file exists
if [ ! -f "$CSV_FILE" ]; then
    echo "Error: CSV file not found at $CSV_FILE"
    exit 1
fi

# Create sanitized model name for directory and filename (replace / with _)
MODEL_SANITIZED=$(echo "$MODEL" | sed 's/\//_/g')

# Create benchmark_results directory structure
RESULT_BASE_DIR="${SCRIPT_DIR}/benchmark_results"
RESULT_MODEL_DIR="${RESULT_BASE_DIR}/${MODEL_SANITIZED}"
mkdir -p "$RESULT_MODEL_DIR"

echo "========================================"
echo "Benchmark Configuration:"
echo "  Model: $MODEL"
echo "  Base URL: $BASE_URL"
echo "  Backend: $BACKEND"
echo "  Device: $DEVICE"
echo "  CSV File: $CSV_FILE"
echo "  Results Directory: $RESULT_MODEL_DIR"
echo "========================================"
echo ""

# Read CSV file and skip header
tail -n +2 "$CSV_FILE" | while IFS=, read -r ISL OSL MAX_CONCURRENCY NUM_PROMPTS; do
    # Generate timestamp
    TIMESTAMP=$(date +"%Y-%m-%d_%H-%M-%S")
    
    # Construct result filename following the pattern expected by summary_report.py
    # Pattern: benchmark_{model}_{device}_{timestamp}_isl-{isl}_osl-{osl}_maxcon-{maxcon}_n-{n}.json
    RESULT_FILENAME="benchmark_${MODEL_SANITIZED}_${DEVICE}_${TIMESTAMP}_isl-${ISL}_osl-${OSL}_maxcon-${MAX_CONCURRENCY}_n-${NUM_PROMPTS}.json"
    
    echo "========================================"
    echo "Running benchmark with parameters:"
    echo "  Input Length (ISL): $ISL"
    echo "  Output Length (OSL): $OSL"
    echo "  Max Concurrency: $MAX_CONCURRENCY"
    echo "  Num Prompts: $NUM_PROMPTS"
    echo "  Result File: $RESULT_FILENAME"
    echo "========================================"
    
    # Run the benchmark command
    python3 benchmark_serving.py \
        --model "$MODEL" \
        --backend "$BACKEND" \
        --base-url "$BASE_URL" \
        --dataset-name random \
        --random-input-len "$ISL" \
        --random-output-len "$OSL" \
        --num-prompts "$NUM_PROMPTS" \
        --max-concurrency "$MAX_CONCURRENCY" \
        --request-rate inf \
        --ignore-eos \
        --save-result \
        --percentile-metrics ttft,tpot,itl,e2el \
        --result-dir "$RESULT_MODEL_DIR" \
        --result-filename "$RESULT_FILENAME"
    
    echo "Completed benchmark for ISL=$ISL, OSL=$OSL, Concurrency=$MAX_CONCURRENCY, NumPrompts=$NUM_PROMPTS"
    echo ""
    
    # Add a small delay between runs to avoid timestamp collisions
    sleep 2
done

echo "========================================"
echo "All benchmarks completed!"
echo "Results saved to: $RESULT_MODEL_DIR"
echo ""
echo "To generate summary report, run:"
echo "python3 summary_report.py $RESULT_MODEL_DIR/benchmark_*.json --output-dir $RESULT_MODEL_DIR"
echo "========================================"

