#!/bin/bash 
#SBATCH -o slurm_batch_infer_Inf_ed_MORE_3x30000_20260402_%a.out
#SBATCH -e slurm_batch_infer_Inf_ed_MORE_3x30000_20260402_%a.err
#SBATCH -t 1-00:00:00
#SBATCH -p youlab-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=32G
#SBATCH --array=1-30
#SBATCH --mail-type=ALL
source activate pytorch_PA_patternprediction
cd /hpc/dctrl/ks723/Physics_constrained_DL_pattern_prediction/sim_to_exp_diffusion/controlnet_essential
export TQDM_DISABLE=1
python batch_infer_Inf_ed_MORE_3x30000.py "$@"
