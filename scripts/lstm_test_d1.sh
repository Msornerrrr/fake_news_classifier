#!/bin/bash

# Define log file location
log_dir="./logs"
log_file="$log_dir/script_$(date '+%Y-%m-%d_%H-%M-%S').log"

# Ensure log directory exists
mkdir -p $log_dir

# Redirect all output to log file
exec > >(tee -a "$log_file") 2>&1

# Start of your script
echo "Script started on $(date)"

# Running Script

# Define the script name
lstm_run="python lstm.py run"

echo "Start running LSTM models on dataset 1..."
for model in LSTM_2023-12-10_05-34-45 LSTM_2023-12-10_05-52-46 LSTM_2023-12-10_06-09-49 LSTM_2023-12-10_06-27-10 LSTM_2023-12-10_06-40-46 LSTM_2023-12-10_06-58-30; do
    echo "Running model $model..."
    $lstm_run -m $model -d 1 || { echo "Running model $model failed"; exit 1; }
    echo
done

echo "Start running LSTM models on dataset 2..."
$lstm_run -m LSTM_2023-12-10_07-08-39 -d 2 || { echo "Running model 2023-12-10_07-08-39 failed"; exit 1; }
