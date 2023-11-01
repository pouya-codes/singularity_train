#!/bin/bash
#SBATCH --job-name writereadme
#SBATCH --cpus-per-task 1
#SBATCH --output /projects/ovcare/classification/cchen/ml/slurm/writereadme.%j.out
#SBATCH --error  /projects/ovcare/classification/cchen/ml/slurm/writereadme.%j.out
#SBATCH -w {w}
#SBATCH -p {p}
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --chdir /projects/ovcare/classification/cchen/ml/docker_train

source /projects/ovcare/classification/cchen/{pyenv}

cd /home/cochen/cchen/ml/docker_train

python app.py -h >> README.md

