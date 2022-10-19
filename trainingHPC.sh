#!/bin/bash
#SBATCH --job-name=pipeline_newgen
#SBATCH --output=my-output.log
#SBATCH --mem=64G
#SBATCH --time=1-10:00:00
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --mail-user=tiago.alves@aicos.fraunhofer.pt
#SBATCH --mail-type=ALL

# mkdir -p /hpc/scratch/$user
# load modules
module load cuda11.2/toolkit/11.2.2

conda activate ldm


conda install -c conda-forge huggingface_hub
 

# cd ~/projects/diffusion/stable-diffusion/ || return
# run your code (pip install modules on login node; datasets read directly from /net/sharedfolders/datasets)
# dvc repro
# python "scripts/txt2img.py "--prompt "Kangaroo dressed in an orange hoodie wearing blue sunglasses in front of the Sidney's Opera House" --plms --ckpt sd-v1-4.ckpt --skip_grid --n_samples 2 --ddim_steps 100
python scripts/stable_txt2img.py   --ddim_eta 0.0   --n_samples 10   --n_iter 4   --scale 7.0   --ddim_steps 50   --ckpt "./trained_models/2022-10-19T16-57-15_HSIL_203_training_images_10000_max_training_steps_firstNameLastName_token_cell_class_word.ckpt"   --prompt "carcinoma cell"')
# copy results back to your home
# cp -r /hpc/scratch/$user/my-results ~oepenna person as a masterpiece portrait painting by John Singer Sargent in the style of Rembrandt/my-results

# delete scratch
# rm -rf /hpc/scratch/$user














