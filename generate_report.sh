#!/bin/bash

# Script to generate summary report from benchmark results
# Usage: ./generate_report.sh <model_name>
# Example: ./generate_report.sh Qwen/Qwen2.5-3B-Instruct

set -e  # Exit on error

# Check if model name is provided
if [ $# -lt 1 ]; then
    echo "Usage: $0 <model_name>"
    echo "Example: $0 Qwen/Qwen2.5-3B-Instruct"
    exit 1
fi

MODEL="$1"

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Create sanitized model name for directory (replace / with _)
MODEL_SANITIZED=$(echo "$MODEL" | sed 's/\//_/g')

# Define result directory
RESULT_BASE_DIR="${SCRIPT_DIR}/benchmark_results"
RESULT_MODEL_DIR="${RESULT_BASE_DIR}/${MODEL_SANITIZED}"

# Check if result directory exists
if [ ! -d "$RESULT_MODEL_DIR" ]; then
    echo "Error: Results directory not found at $RESULT_MODEL_DIR"
    echo "Have you run the benchmarks yet?"
    exit 1
fi

# Check if there are any benchmark JSON files
if ! ls "$RESULT_MODEL_DIR"/benchmark_*.json 1> /dev/null 2>&1; then
    echo "Error: No benchmark JSON files found in $RESULT_MODEL_DIR"
    echo "Have you run the benchmarks yet?"
    exit 1
fi

# Generate timestamp for report ID
TIMESTAMP=$(date +"%Y-%m-%d_%H-%M-%S")
REPORT_ID="${MODEL_SANITIZED}_${TIMESTAMP}"

echo "========================================"
echo "Generating Summary Report"
echo "  Model: $MODEL"
echo "  Results Directory: $RESULT_MODEL_DIR"
echo "  Report ID: $REPORT_ID"
echo "========================================"
echo ""

# Run summary report generation
python3 "${SCRIPT_DIR}/summary_report.py" \
    "$RESULT_MODEL_DIR"/benchmark_*.json \
    --output-dir "$RESULT_MODEL_DIR" \
    --pattern "benchmark_*.json"

echo ""
echo "========================================"
echo "Summary report generated successfully!"
echo "Check the following files:"
echo "  - Markdown report: $RESULT_MODEL_DIR/benchmark_display_*.md"
echo "  - CSV data: $RESULT_MODEL_DIR/data/benchmark_stats_*.csv"
echo "========================================"

