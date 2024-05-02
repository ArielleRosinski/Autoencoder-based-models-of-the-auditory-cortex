#!/bin/bash
#SBATCH --nodes=1
#SBATCH --mem=100G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --array=1-65
#SBATCH --job-name=ae_run
#SBATCH --output=logs/ae_%A_%a.out
#SBATCH --exclude=gpu-sr670-20


config=/nfs/nhome/live/arosinski/msc_project_files/AE_reg_params_31_7_2023.txt

regularizer=$(awk -v ArrayTaskID=$SLURM_ARRAY_TASK_ID '$1==ArrayTaskID {print $2}' $config)

reg_lambda=$(awk -v ArrayTaskID=$SLURM_ARRAY_TASK_ID '$1==ArrayTaskID {print $3}' $config)

foldid=$(awk -v ArrayTaskID=$SLURM_ARRAY_TASK_ID '$1==ArrayTaskID {print $4}' $config)

hidden_units=$(awk -v ArrayTaskID=$SLURM_ARRAY_TASK_ID '$1==ArrayTaskID {print $5}' $config)

exp_decay=$(awk -v ArrayTaskID=$SLURM_ARRAY_TASK_ID '$1==ArrayTaskID {print $6}' $config)

folder_name=$(awk -v ArrayTaskID=$SLURM_ARRAY_TASK_ID '$1==ArrayTaskID {print $7}' $config)

time_lag=$(awk -v ArrayTaskID=$SLURM_ARRAY_TASK_ID '$1==ArrayTaskID {print $8}' $config)

module load python
python /nfs/nhome/live/arosinski/git_project_folder/AutoEncoder/model_training.py $regularizer $reg_lambda $foldid $hidden_units $exp_decay $folder_name $time_lag 

