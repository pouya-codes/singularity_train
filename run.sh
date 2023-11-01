#!/bin/bash
#SBATCH --job-name train
#SBATCH --cpus-per-task 4
#SBATCH --output2 /home/poahmadvand/ml/slurm/train_%j.out
#SBATCH --error /home/poahmadvand/ml/slurm/train_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=pouya.ahmadvand@gmail.com
#SBATCH -w dlhost04
#SBATCH -p rtx5000
#SBATCH --gres=gpu:1
#SBATCH --time=01:30:00
#SBATCH --chdir /projects/ovcare/classification/singularity_modules/singularity_train
#SBATCH --mem=15G

/opt/singularity-3.4.0/bin/singularity run --bind /projects:/projects --nv singularity_train.sif from-experiment-manifest "sample_manifest.yaml"