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
lstm_tune="python lstm.py tune"

# LSTM
echo "Start finetuning lstm..."
for hyperparameter in dropout_prob use_pretrained_embeddings hidden_dim; do
    echo "Tuning hyperparameter $hyperparameter..."
    $lstm_tune -hp $hyperparameter || { echo "Tuning hyperparameter $hyperparameter failed"; exit 1; }
    echo
done

echo "Start running default lstm model..."
$lstm_run -m "LSTM_2023-12-10_05-34-45" -nt -p || { echo "Running default lstm model failed"; exit 1; }
echo

echo "Start running best lstm model..."
for dataset in 2 1; do
    echo "Running best lstm model on dataset $dataset..."
    $lstm_run -m "LSTM_2023-12-10_06-58-30" -d $dataset || { echo "Running best lstm model failed"; exit 1; }
    echo
done
