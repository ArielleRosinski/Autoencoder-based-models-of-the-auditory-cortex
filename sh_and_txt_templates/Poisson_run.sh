#!/bin/bash
#SBATCH --job-name=pois
#SBATCH --array=1-816
#SBATCH --nodes=1
#SBATCH --mem=100G
#SBATCH --partition=cpu
#SBATCH --output=logs_kfold/nn_poisson_%A_%a.out


config=/nfs/nhome/live/arosinski/msc_project_files/poisson_AE_params.txt

model_path=$(awk -v ArrayTaskID=$SLURM_ARRAY_TASK_ID '$1==ArrayTaskID {print $2}' $config)

resp=$(awk -v ArrayTaskID=$SLURM_ARRAY_TASK_ID '$1==ArrayTaskID {print $3}' $config)

resp_val=$(awk -v ArrayTaskID=$SLURM_ARRAY_TASK_ID '$1==ArrayTaskID {print $4}' $config)

ni=$(awk -v ArrayTaskID=$SLURM_ARRAY_TASK_ID '$1==ArrayTaskID {print $5}' $config)

no=$(awk -v ArrayTaskID=$SLURM_ARRAY_TASK_ID '$1==ArrayTaskID {print $6}' $config)

nh_list=$(awk -v ArrayTaskID=$SLURM_ARRAY_TASK_ID '$1==ArrayTaskID {print $7}' $config)

cell_id=$(awk -v ArrayTaskID=$SLURM_ARRAY_TASK_ID '$1==ArrayTaskID {print $8}' $config)

folder_name=$(awk -v ArrayTaskID=$SLURM_ARRAY_TASK_ID '$1==ArrayTaskID {print $9}' $config)

reg_lambda=$(awk -v ArrayTaskID=$SLURM_ARRAY_TASK_ID '$1==ArrayTaskID {print $10}' $config)

module load python 
python /nfs/nhome/live/arosinski/git_project_folder/AutoEncoder/poisson_NN_regression_model.py $model_path $resp $resp_val $ni $no $nh_list $cell_id $folder_name $reg_lambda

