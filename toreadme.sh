echo """# Train

### Development Information ###

\`\`\`
Date Created: 22 July 2020
Last Update: 11 April 2021 by Amirali
Developer: Colin Chen
Version: 1.0
\`\`\`

**Before running any experiment to be sure you are using the latest commits of all modules run the following script:**
\`\`\`
(cd /projects/ovcare/classification/singularity_modules ; ./update_modules.sh --bcgsc-pass your/bcgsc/path)
\`\`\`

### Usage ###
\`\`\`
""" > README.md

python app.py -h >> README.md
echo >> README.md
python app.py from-experiment-manifest -h >> README.md
echo >> README.md
python app.py from-arguments -h >> README.md
echo >> README.md
python app.py from-arguments freeze_training -h >> README.md
echo >> README.md
echo """\`\`\`
""" >> README.md
