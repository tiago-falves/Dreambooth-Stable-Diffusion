

# ## Training
# 
# If training a person or subject, keep an eye on your project's `logs/{folder}/images/train/samples_scaled_gs-00xxxx` generations.
# 
# If training a style, keep an eye on your project's `logs/{folder}/images/train/samples_gs-00xxxx` generations.

# In[8]:


# Training

# This isn't used for training, just to help you remember what your trained into the model.
project_name = "HSIL"

# MAX STEPS
# How many steps do you want to train for?
max_training_steps = 1000

# Match class_word to the category of the regularization images you chose above.
class_word = "cell" # typical uses are "man", "person", "woman"

# This is the unique token you are incorporating into the stable diffusion model.
token = "firstNameLastName"


reg_data_root = "/workspace/Dreambooth-Stable-Diffusion/regularization_images/" + "cell"

get_ipython().system('rm -rf training_images/.ipynb_checkpoints')
get_ipython().system('python "main.py"   --base configs/stable-diffusion/v1-finetune_unfrozen.yaml   -t   --actual_resume "model.ckpt"   --reg_data_root "{reg_data_root}"   -n "{project_name}"   --gpus 0,   --data_root "/workspace/Dreambooth-Stable-Diffusion/training_images"   --max_training_steps {max_training_steps}   --class_word "{class_word}"   --token "{token}"   --no-test')




# ## Copy and name the checkpoint file

# In[ ]:


# Copy the checkpoint into our `trained_models` folder

directory_paths = get_ipython().getoutput('ls -d Dreambooth-Stable-Diffusion/logs/*')
last_checkpoint_file = directory_paths[-1] + "/checkpoints/last.ckpt"
training_images = get_ipython().getoutput('find training_images/*')
date_string = get_ipython().getoutput('date +"%Y-%m-%dT%H-%M-%S"')
file_name = date_string[-1] + "_" + project_name + "_" + str(len(training_images)) + "_training_images_" +  str(max_training_steps) + "_max_training_steps_" + token + "_token_" + class_word + "_class_word.ckpt"

file_name = file_name.replace(" ", "_")

get_ipython().system('mkdir -p trained_models')
get_ipython().system('mv "{last_checkpoint_file}" "trained_models/{file_name}"')

print("Download your trained model file from trained_models/" + file_name + " and use in your favorite Stable Diffusion repo!")


# # Optional - Upload to google drive
# * run the following commands in a new `terminal` in the `Dreambooth-Stable-Diffusion` directory
# * `chmod +x ./gdrive`
# * `./gdrive about`
# * `paste your token here after navigating to the link`
# * `./gdrive upload trained_models/{file_name.ckpt}`

# # Big Important Note!
# 
# The way to use your token is `<token> <class>` ie `joepenna person` and not just `joepenna`

# ## Generate Images With Your Trained Model!

# In[ ]:


# get_ipython().system('python Dreambooth-Stable-Diffusion/scripts/stable_txt2img.py   --ddim_eta 0.0   --n_samples 1   --n_iter 4   --scale 7.0   --ddim_steps 50   --ckpt "/workspace/Dreambooth-Stable-Diffusion/trained_models/{file_name}"   --prompt "joepenna person as a masterpiece portrait painting by John Singer Sargent in the style of Rembrandt"')

