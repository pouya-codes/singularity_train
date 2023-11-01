#!/bin/bash
#SBATCH --cpus-per-task 1
#SBATCH --mail-type=END,FAIL          # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=pouya.ahmadvand@gmail.com     # Where to send mail
#SBATCH --gres=gpu:1
#SBATCH --time=4-90:00:00
#SBATCH --chdir /home/poahmadvand/share/Sub_Types/04_Train_Model/singularity_train
#SBATCH --mem=20G

source /home/poahmadvand/py3env/bin/activate
SPLIT=split${SPLIT_NUMBER}
echo $SPLIT
python app.py from-experiment-manifest ../tumor_grade/tumor_grade.yaml --component_id $SPLIT