#!/bin/bash
# scripts/run_benchmark.sh
# Execute the retrieval benchmark outside the IDE runtime, preventing macOS sleep.

WORKSPACE_DIR="/Users/shivasairahulkuntala/Developer/AI_Projects/nlp_models/Harrison"
cd "$WORKSPACE_DIR"

# Enforce environment variables to avoid OpenMP threading conflict crashes
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

echo "=== Starting Harrison Retrieval Benchmark ==="
echo "Preventing system sleep with 'caffeinate'..."

# Run under python 3.12 virtual environment, wrapped in caffeinate to prevent sleep
caffeinate -i .venv312/bin/python scripts/benchmark_retrieval_safe.py

echo "=== Benchmark execution finished! ==="
