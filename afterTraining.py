
# ## Copy and name the checkpoint file

# In[ ]:

# Training

# This isn't used for training, just to help you remember what your trained into the model.
project_name = "HSIL"

# MAX STEPS
# How many steps do you want to train for?
max_training_steps = 5

# Match class_word to the category of the regularization images you chose above.
class_word = "cell" # typical uses are "man", "person", "woman"

# This is the unique token you are incorporating into the stable diffusion model.
token = "firstNameLastName"


reg_data_root = "/workspace/Dreambooth-Stable-Diffusion/regularization_images/" + "cell"

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