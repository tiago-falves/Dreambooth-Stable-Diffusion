#!/usr/bin/env python
# coding: utf-8

# # Dreambooth
# ### Notebook implementation by Joe Penna (@MysteryGuitarM on Twitter) - Improvements by David Bielejeski
# 
# ### Instructions
# - Sign up for RunPod here: https://runpod.io/?ref=n8yfwyum
#     - Note: That's my personal referral link. Please don't use it if we are mortal enemies.
# 
# - Click *Deploy* on either `SECURE CLOUD` or `COMMUNITY CLOUD`
# 
# - Follow the rest of the instructions in this video: https://www.youtube.com/watch?v=7m__xadX0z0#t=5m33.1s
# 
# Latest information on:
# https://github.com/JoePenna/Dreambooth-Stable-Diffusion

# ## Build Environment

# In[ ]:


# If running on Vast.AI, copy the code in this cell into a new notebook. Run it, then launch the `dreambooth_runpod_joepenna.ipynb` notebook from the jupyter interface.
# get_ipython().system('git clone https://github.com/JoePenna/Dreambooth-Stable-Diffusion')


# In[1]:


# BUILD ENV
get_ipython().system('pip install omegaconf')
get_ipython().system('pip install einops')
get_ipython().system('pip install pytorch-lightning==1.6.5')
get_ipython().system('pip install test-tube')
get_ipython().system('pip install transformers')
get_ipython().system('pip install kornia')
get_ipython().system('pip install -e git+https://github.com/CompVis/taming-transformers.git@master#egg=taming-transformers')
get_ipython().system('pip install -e git+https://github.com/openai/CLIP.git@main#egg=clip')
get_ipython().system('pip install setuptools==59.5.0')
get_ipython().system('pip install pillow==9.0.1')
get_ipython().system('pip install torchmetrics==0.6.0')
get_ipython().system('pip install -e .')
get_ipython().system('pip install protobuf==3.20.1')
get_ipython().system('pip install gdown')
get_ipython().system('pip install -qq diffusers["training"]==0.3.0 transformers ftfy')
get_ipython().system('pip install -qq "ipywidgets>=7,<8"')
get_ipython().system('pip install huggingface_hub')
get_ipython().system('pip install ipywidgets==7.7.1')


# In[2]:


# Hugging Face Login
#from huggingface_hub import notebook_login

#notebook_login()

from huggingface_hub.hf_api import HfFolder
HfFolder.save_token('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')


# In[3]:


# # Download the 1.4 sd model
from IPython.display import clear_output

from huggingface_hub import hf_hub_download
downloaded_model_path = hf_hub_download(
 repo_id="CompVis/stable-diffusion-v-1-4-original",
 filename="sd-v1-4.ckpt",
 use_auth_token=True
)


# Move the sd-v1-4.ckpt to the root of this directory as "model.ckpt"
actual_locations_of_model_blob = get_ipython().getoutput('readlink -f {downloaded_model_path}')
get_ipython().system('mv {actual_locations_of_model_blob[-1]} model.ckpt')
clear_output()
print("✅ model.ckpt successfully downloaded")



# # Upload your training images
# Upload 10-20 images of someone to
# 
# ```
# /workspace/Dreambooth-Stable-Diffusion/training_images
# ```
# 
# WARNING: Be sure to upload an *even* amount of images, otherwise the training inexplicably stops at 1500 steps.
# 
# *   2-3 full body
# *   3-5 upper body
# *   5-12 close-up on face
# 
# The images should be:
# 
# - as close as possible to the kind of images you're trying to make

# In[2]:


#@markdown Add here the URLs to the images of the subject you are adding
#Carcinoma
urls = [
"https://i.imgur.com/mVT8iRP.png",
"https://i.imgur.com/WYr3yMv.png",
"https://i.imgur.com/8A7Iv46.png",
"https://i.imgur.com/x0kDidX.png",
"https://i.imgur.com/qgUHyeL.png",
"https://i.imgur.com/VDBTtzK.png",
"https://i.imgur.com/crBaLt0.png",
"https://i.imgur.com/xRVxMUg.png",
"https://i.imgur.com/GHhNDsQ.png",
"https://i.imgur.com/2EkdNKK.png",
"https://i.imgur.com/PCr3KgX.png",
"https://i.imgur.com/LGfeZW5.png",
"https://i.imgur.com/LLW4N3K.png",

 # You can add additional images here -- about 20-30 images in different
]



#@title Download and check the images you have just added
import os
import requests
from io import BytesIO
from PIL import Image


def image_grid(imgs, rows, cols):
 assert len(imgs) == rows*cols

 w, h = imgs[0].size
 grid = Image.new('RGB', size=(cols*w, rows*h))
 grid_w, grid_h = grid.size

 for i, img in enumerate(imgs):
  grid.paste(img, box=(i%cols*w, i//cols*h))
 return grid

def download_image(url):
 try:
  response = requests.get(url)
 except:
  return None
 return Image.open(BytesIO(response.content)).convert("RGB")

images = list(filter(None,[download_image(url) for url in urls]))
save_path = "./training_images/carcinoma"
if not os.path.exists(save_path):
 os.mkdir(save_path)
[image.save(f"{save_path}/{i}.png", format="png") for i, image in enumerate(images)]
image_grid(images, 1, len(images))


# In[ ]:





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
max_training_steps = 5

# Match class_word to the category of the regularization images you chose above.
class_word = "cell" # typical uses are "man", "person", "woman"

# This is the unique token you are incorporating into the stable diffusion model.
token = "firstNameLastName"


reg_data_root = "./regularization_images/" + "cell"

get_ipython().system('rm -rf training_images/carcinoma/.ipynb_checkpoints')

# Opening a file
file1 = open('args.txt', 'w')
L = [reg_data_root +"\n", project_name]
s = "Hello\n"
  
# Writing multiple strings
# at a time
file1.writelines(L)



