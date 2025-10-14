#!/bin/bash

# Assign the arguments to variables
input_dir=$1
ID=$2
nnUNet_dir=$3
config=$4

# Set environment variables
export nnUNet_results="${nnUNet_dir}/nnUNet_results"
export nnUNet_n_proc_DA=18

# Run the nnUNet prediction
nnUNetv2_predict -d $ID -i "$input_dir" -o "$input_dir" -f all -tr nnUNetTrainer -c "$config" -p nnUNetPlans
    

