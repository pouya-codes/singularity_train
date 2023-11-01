#!/bin/bash
#SBATCH --job-name toreadme
#SBATCH --cpus-per-task 1
#SBATCH --output2 /home/poahmadvand/ml/slurm/classification/toreadme/toreadme.%j.out
#SBATCH --error  /home/poahmadvand/ml/slurm/classification/toreadme/toreadme.%j.out
#SBATCH -w dlhost04
#SBATCH -p rtx5000
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00

echo """# Singularity Train

**Before runing any experiment to be sure you are using the latest commits of all modules run the following script:**
\`\`\`
/projects/ovcare/classification/singularity_modules/update_moudles.sh
\`\`\`
### Development Information ###

\`\`\`
"Date Created: 22 July 2020"
"Last Update:" $(date) "by" $(whoami)
"Developer: Colin Chen"
"Version: 1.0"
\`\`\`
### 1. Description ###
This branch is singularity-based version on [docker_train](https://svn.bcgsc.ca/bitbucket/projects/MLOVCA/repos/docker_train/browse) master branch.
### 2. How to Use ###
Follow steps in this [link](https://www.bcgsc.ca/wiki/display/OVCARE/Singularity+on+Numbers).


To build singularity image

\`\`\`
$singularity build --remote singularity_train.sif Singularityfile.def
\`\`\`

To run the container afterwards

\`\`\`
$singularity run --nv singularity_train.sif from-experiment-manifest "path/to/manifest/file/location" 
\`\`\`

Here's an example of the setup you can use:

\`sample_manifest.yaml\`

### 3. Usage ###
\`\`\`
""" > README.md

python app.py -h >> README.md
echo >> README.md
python app.py from-experiment-manifest -h >> README.md
echo >> README.md
python app.py from-arguments -h >> README.md
echo >> README.md
python app.py from-arguments early_stopping -h >> README.md
echo >> README.md
python app.py from-arguments test_model -h >> README.md
echo >> README.md
echo """\`\`\`
""" >> README.md

