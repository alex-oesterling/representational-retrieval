#!/bin/bash
#SBATCH -c 1
#SBATCH -t 0-3:59
#SBATCH -p seas_gpu
#SBATCH --gres=gpu:tesla_v100-pcie-32gb:4
#SBATCH --mem=128000
#SBATCH --mail-type=ALL

module load python/3.10.12-fasrc01
mamba activate torchy
python MMR_retrieval.py -device "cuda" -dataset "celeba" -n_samples 20000 -query "A photo of a Professor" -k 10 -functionclass "randomforest"
