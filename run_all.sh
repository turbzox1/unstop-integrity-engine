#!/usr/bin/env bash
# Single root command: run the pipeline on the sample data, then score it against ground truth.
set -euo pipefail
python run_integrity_check.py --input_dir ./sample_submissions --output_dir ./reports "$@"
python evaluate.py --reports_dir ./reports --input_dir ./sample_submissions
