#!/bin/bash

# Define log file location
log_dir="./logs"
log_file="$log_dir/basic_$(date '+%Y-%m-%d_%H-%M-%S').log"

# Ensure log directory exists
mkdir -p $log_dir

# Redirect all output to log file
exec > >(tee -a "$log_file") 2>&1

# Start of your script
echo "Script started on $(date)"

# Running Script

# Define the script name
baseline_name="python baseline.py"
naive_bayes_name="python naive_bayes.py"

# Baseline
echo "Start running baseline..."
for dataset in 2 1; do
    for method in "MostCommonWords" "NumPunctuation" "NumCaps"; do
        echo "Running baseline on dataset $dataset with method $method..."
        $baseline_name -d $dataset -m $method || { echo "Baseline with method $method failed"; exit 1; }
        echo
    done
done

# Naive Bayes
echo "Start running naive bayes..."
for dataset in 2 1; do
    echo "Running naive bayes on dataset $dataset..."
    $naive_bayes_name -d $dataset || { echo "Naive bayes failed"; exit 1; }
    echo
done