#!/bin/bash
#SBATCH --mail-user=xg7@nyu.edu
#SBATCH --mail-type=FAIL
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:a10:1
#SBATCH --time=7-00:00:00
#SBATCH --mem=16000
#SBATCH --output=Job.%j.out
#SBATCH --error=Job.%j.err
#SBATCH --account=xg7
#SBATCH --partition=aquila
source activate pytorch
python < main.py
conda deactivate