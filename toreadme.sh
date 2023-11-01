echo """# Train

### Development Information ###

\`\`\`
Date Created: 22 July 2020
Last Update: 15 July 2021 by Pouya
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

python3 app.py -h >> README.md
echo >> README.md
python3 app.py from-experiment-manifest -h >> README.md
echo >> README.md
python3 app.py from-arguments -h >> README.md
echo >> README.md
python3 app.py from-arguments freeze_training -h >> README.md
echo >> README.md
echo """\`\`\`
""" >> README.md

echo """
Note: \`freeze_training\` subparser MUST be used in your manifest. Due to more readability, the parser is
defined in this way (having multiple subparserss instead of just one).
""" >> README.md

echo """
### Config File ###
Model definition and augmentations are defined in the \`config.json\` file. It is divided into:
1. \`model\`:
1.1. \`num_subtypes\`: number of output neurons (number of classes).
1.2. \`base_model\`: the baseline model.
1.3. \`pretrained\`: {"true","false"} -> if true, uses ImageNet trained weights.
1.4. \`last_layers\`: {"short","long"} -> "short" means only changing last layer neurons to be compatible to our dataset, but "long" means adding more layers to fully connected section.
1.5. \`concat_pool\`: {"true","false"} -> if true, uses concatination of max and average pooling. If not, use only max pooling. (Note: it only affects when the last_layer is long!)
2. \`normalize\`:
2.1. \`use_normalize\`: {"true","false"} -> if true, normalize the dataset based on provided mean and std. Otherwise, use 0.5 as mean and std for each channel.
2.2. \`mean\`: average for each channel
2.3. \`std\`: standard deviation for each channel
3. \`augmentation\`:
3.1. \`use_augmentation\`: {"true","false"} -> whether use augmentation or not at all
3.2. \`flip\`: {"true","false"} -> if true, add vertical and horizantal flip to the augmentation list.
3.3. \`color_jitter\`: {"true","false"} -> if true, add color jitter to the augmentation list.
3.3. \`rotation\`: {"true","false"} -> if true, add 20 degree rotation to the augmentation list.
3.3. \`crop\`: {int} -> if set, add crop of determied size to the augmentation list.
3.3. \`resize\`: {int} -> if set, add resize of determied size to the augmentation list.
3.4. \`size_jitter\`:
3.4.1. \`use_size_jitter\`: {"true","false"} -> if true, add size_jitter to the augmentation list.
3.4.2. \`ratio\`: {float} -> ratio of the original image size
3.4.3. \`probability\`: {float} -> probability of doing this augmentation
3.4.4. \`color\`: {"white","black"} -> the color of padding when use ratio less than 1.
3.5. \`cut_out\`:
3.5.1. \`use_num_cut\`: {"true","false"} -> if true, add num_cut to the augmentation list.
3.5.2. \`num_cut\`: {int} -> number of cutouts
3.5.3. \`size_cut\`: {int} -> size of each cutout in pixels
3.5.4. \`color_cut\`: {"white","black"} -> the color of cutouts
4. \`use_weighted_loss\`:
4.1. \`use_weighted_loss\`: {"true","false"} -> if true, use weighted loss.
4.2. \`weight\`: {array of floats} -> weights used for each class in the loss
5. \`use_weighted_sampler\`: {"true","false"} -> if true, use a weighted sampler that sample the data with the weights 1/num_class (for imbalancing).
6. \`use_balanced_sampler\`: {"true","false"} -> if true, use a balanced sampler that in each batch, we will have same number of data from each class (for imbalancing).
7. \`mix_up\`:
7.1. \`use_mix_up\`: {"true","false"} -> if true, use mix_up technique.
7.2. \`alpha\`: {float} -> betha distribution (recommended to set as 0.4)
8. \`freeze\`: useless (TODO: remove this flag)
9. \`continue_train\`: useless (TODO: remove this flag)
10. \`optimizer\`: Optimizer settings ..
11. \`scheduler\`: Scheduler settings ..

Note: If both resize and crop are added, first resize is applied and then crop.
""" >> README.md
