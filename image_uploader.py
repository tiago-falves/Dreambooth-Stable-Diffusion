
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
save_path = "./training_images/really_carcinoma"
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
  
# Writing multiple strings
# at a time
file1.writelines(L)



