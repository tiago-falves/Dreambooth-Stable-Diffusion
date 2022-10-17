#!/bin/bash
#SBATCH --job-name=pipeline_newgen
#SBATCH --output=my-output.log
#SBATCH --mem=64G
#SBATCH --time=7-24:00:00
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --mail-user=tiago.falves98@gmail.com
#SBATCH --mail-type=ALL

# mkdir -p /hpc/scratch/$user
# load modules
module load cuda11.2/toolkit/11.2.2

conda env create --file environment.yaml
conda activate ldm

conda install ipython3
conda install -c conda-forge huggingface_hub

ipython3 dreambooth_runpod_joepenna.py 

# cd ~/projects/diffusion/stable-diffusion/ || return
# run your code (pip install modules on login node; datasets read directly from /net/sharedfolders/datasets)
# dvc repro
# python scripts/txt2img.py --prompt "Kangaroo dressed in an orange hoodie wearing blue sunglasses in front of the Sidney's Opera House" --plms --ckpt sd-v1-4.ckpt --skip_grid --n_samples 2 --ddim_steps 100

# copy results back to your home
# cp -r /hpc/scratch/$user/my-results ~/my-results

# delete scratch
# rm -rf /hpc/scratch/$user














