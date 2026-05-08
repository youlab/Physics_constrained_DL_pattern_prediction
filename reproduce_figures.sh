#!/bin/bash 
#SBATCH -o slurm_reproduce_figures_%j.out
#SBATCH -e slurm_reproduce_figures_%j.err
#SBATCH -p youlab-gpu
#SBATCH --time=02:00:00
#SBATCH --exclusive
#SBATCH --mem=24G
#SBATCH --mail-type=ALL

source ~/.bashrc  
conda activate pytorch_PA_patternprediction
python reproduce_figures.py 