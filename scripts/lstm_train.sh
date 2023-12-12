#!/bin/bash

# Define log file location
log_dir="./logs"
log_file="$log_dir/lstm_train_$(date '+%Y-%m-%d_%H-%M-%S').log"

# Ensure log directory exists
mkdir -p $log_dir

# Redirect all output to log file
exec > >(tee -a "$log_file") 2>&1

# Start of your script
echo "Script started on $(date)"

# LSTM Training Script

# Define the script name
script_name="python lstm.py train"

# Run basic training command
echo "Running basic training..."
$script_name -s || { echo "Basic training failed"; exit 1; }

# Dropout variations
echo "Training with different dropout rates..."
for dropout in 0.1 0.5; do
    $script_name -dp $dropout -s || { echo "Training with dropout $dropout failed"; exit 1; }
done

# Pretrained embedding
echo "Training with pretrained embedding..."
$script_name -pe -s || { echo "Training with pretrained embedding failed"; exit 1; }

echo "Training with different hidden dimensions..."
for hidden_dim in 128 512; do
    $script_name -hd $hidden_dim -s || { echo "Training with hidden dimension $hidden_dim failed"; exit 1; }
done

# Dataset variation
echo "Training with different dataset..."
$script_name -d 1 -s || { echo "Training with dataset 1 failed"; exit 1; }

echo "All training commands executed successfully."
