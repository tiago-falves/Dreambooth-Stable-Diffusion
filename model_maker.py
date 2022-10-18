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




# In[2]:


# Hugging Face Login
from huggingface_hub.hf_api import HfFolder
HfFolder.save_token('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')



from huggingface_hub import hf_hub_download
downloaded_model_path = hf_hub_download(
 repo_id="CompVis/stable-diffusion-v-1-4-original",
 filename="sd-v1-4.ckpt",
 use_auth_token=True
)
print("Downloaded model path:")

# Move the sd-v1-4.ckpt to the root of this directory as "model.ckpt"
actual_locations_of_model_blob = get_ipython().getoutput('readlink -f {downloaded_model_path}')
get_ipython().system('mv {actual_locations_of_model_blob[-1]} model.ckpt')
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
urls = [
"https://i.imgur.com/Zrhc1xn.png",
"https://i.imgur.com/rQs39ze.png",
"https://i.imgur.com/X1xh4t3.png",
"https://i.imgur.com/HKXsVrv.png",
"https://i.imgur.com/KqwPiNj.png",
"https://i.imgur.com/dHzV5lL.png",
"https://i.imgur.com/XiSnb1d.png",
"https://i.imgur.com/snRWwLy.png",
"https://i.imgur.com/ndalRcO.png",
"https://i.imgur.com/8GQUISI.png",
"https://i.imgur.com/lXythFt.png",
"https://i.imgur.com/Wp1QTqY.png",
"https://i.imgur.com/pjcNhaA.png",
"https://i.imgur.com/om5jn95.png",
"https://i.imgur.com/ckGYFW9.png",
"https://i.imgur.com/AYUbPsm.png",
"https://i.imgur.com/YMeXkYw.png",
"https://i.imgur.com/P0Oj42Z.png",
"https://i.imgur.com/Gk17ikS.png",
"https://i.imgur.com/VsH7DpB.png",
"https://i.imgur.com/oOlMaDO.png",
"https://i.imgur.com/Zjy0yfQ.png",
"https://i.imgur.com/IDvEZuY.png",
"https://i.imgur.com/z9gaM3z.png",
"https://i.imgur.com/NqWYcdf.png",
"https://i.imgur.com/gH9R8sJ.png",
"https://i.imgur.com/8Ev9Vmn.png",
"https://i.imgur.com/OemfawQ.png",
"https://i.imgur.com/NQn3Io4.png",
"https://i.imgur.com/Hz0kVWT.png",
"https://i.imgur.com/8LkeYI3.png",
"https://i.imgur.com/jKCZ48d.png",
"https://i.imgur.com/i93D4Kl.png",
"https://i.imgur.com/bAwqndO.png",
"https://i.imgur.com/LNc1vMs.png",
"https://i.imgur.com/OrgvYrL.png",
"https://i.imgur.com/jxdLCm5.png",
"https://i.imgur.com/V0DnTYs.png",
"https://i.imgur.com/CzXMjAY.png",
"https://i.imgur.com/sdJiMnS.png",
"https://i.imgur.com/RjArzCh.png",
"https://i.imgur.com/kRmG5MJ.png",
"https://i.imgur.com/N7xvOAk.png",
"https://i.imgur.com/yJ63nJv.png",
"https://i.imgur.com/r7cbuDl.png",
"https://i.imgur.com/DwIi8rF.png",
"https://i.imgur.com/fGiRaCP.png",
"https://i.imgur.com/VEV8d0d.png",
"https://i.imgur.com/D86KoFB.png",
"https://i.imgur.com/v26qeoX.png",
"https://i.imgur.com/OxUVchl.png",
"https://i.imgur.com/9355Xgd.png",
"https://i.imgur.com/9jHgugR.png",
"https://i.imgur.com/Jr8MvD3.png",
"https://i.imgur.com/IZwOup0.png",
"https://i.imgur.com/OUES8XA.png",
"https://i.imgur.com/tGI2ZhE.png",
"https://i.imgur.com/vBxXKqT.png",
"https://i.imgur.com/uHbwVkA.png",
"https://i.imgur.com/v4pqJ4u.png",
"https://i.imgur.com/sq4DrQV.png",
"https://i.imgur.com/vO7yH7E.png",
"https://i.imgur.com/OGNeqxW.png",
"https://i.imgur.com/tzg0rjM.png",
"https://i.imgur.com/FkRTlNS.png",
"https://i.imgur.com/AAilytJ.png",
"https://i.imgur.com/ADEH7mP.png",
"https://i.imgur.com/y9PLcDN.png",
"https://i.imgur.com/JVzjCCC.png",
"https://i.imgur.com/6UnkhBz.png",
"https://i.imgur.com/atSeS3j.png",
"https://i.imgur.com/uJ9F3wV.png",
"https://i.imgur.com/QaROpp2.png",
"https://i.imgur.com/fmuIpq3.png",
"https://i.imgur.com/22XVc6m.png",
"https://i.imgur.com/0lRgTwG.png",
"https://i.imgur.com/DfzSHBz.png",
"https://i.imgur.com/VkUs5Et.png",
"https://i.imgur.com/UUUw3OC.png",
"https://i.imgur.com/JrX3qUY.png",
"https://i.imgur.com/SaQwycM.png",
"https://i.imgur.com/eWGxxDP.png",
"https://i.imgur.com/itKVj6M.png",
"https://i.imgur.com/8QA1mhF.png",
"https://i.imgur.com/K95rUPz.png",
"https://i.imgur.com/rgEXw4e.png",
"https://i.imgur.com/8Oe9s0A.png",
"https://i.imgur.com/443s2YS.png",
"https://i.imgur.com/j1kKZnq.png",
"https://i.imgur.com/nMo1eXU.png",
"https://i.imgur.com/qYSnPDF.png",
"https://i.imgur.com/SRHVoqQ.png",
"https://i.imgur.com/GKpxm1Z.png",
"https://i.imgur.com/1EEBTpQ.png",
"https://i.imgur.com/VhjfGQv.png",
"https://i.imgur.com/gntnIFu.png",
"https://i.imgur.com/W7ciGGK.png",
"https://i.imgur.com/GrbcVLX.png",
"https://i.imgur.com/wdiyPnI.png",
"https://i.imgur.com/PSOSrLk.png",
"https://i.imgur.com/sbz02Ju.png",
"https://i.imgur.com/YNwhPhT.png",
"https://i.imgur.com/kEZj3Bo.png",
"https://i.imgur.com/Q6QbgJe.png",
"https://i.imgur.com/La8nSIR.png",
"https://i.imgur.com/xWhdLcK.png",
"https://i.imgur.com/BWNmi36.png",
"https://i.imgur.com/4DN26sr.png",
"https://i.imgur.com/GTrL9Vr.png",
"https://i.imgur.com/c3oIMVk.png",
"https://i.imgur.com/pesxGtW.png",
"https://i.imgur.com/diQ9W7v.png",
"https://i.imgur.com/GvwibRN.png",
"https://i.imgur.com/0FJhYhX.png",
"https://i.imgur.com/XU1Zt5l.png",
"https://i.imgur.com/feaFfc0.png",
"https://i.imgur.com/sYsJe9S.png",
"https://i.imgur.com/NueMhcL.png",
"https://i.imgur.com/CDpQlqD.png",
"https://i.imgur.com/yzg97b3.png",
"https://i.imgur.com/EAEdnyV.png",
"https://i.imgur.com/vcfkUuT.png",
"https://i.imgur.com/xsLXkZH.png",
"https://i.imgur.com/ppWaHb9.png",
"https://i.imgur.com/UoHbFLe.png",
"https://i.imgur.com/zzuKQ90.png",
"https://i.imgur.com/nvx82Wr.png",
"https://i.imgur.com/X51KSTZ.png",
"https://i.imgur.com/AwEDCdp.png",
"https://i.imgur.com/z8X68OE.png",
"https://i.imgur.com/h17sJED.png",
"https://i.imgur.com/doTic44.png",
"https://i.imgur.com/ev0sst6.png",
"https://i.imgur.com/IznLA1D.png",
"https://i.imgur.com/xuDQmkz.png",
"https://i.imgur.com/6BMeiGx.png",
"https://i.imgur.com/Aq5Ymen.png",
"https://i.imgur.com/dxDP9Cp.png",
"https://i.imgur.com/R4yJoGv.png",
"https://i.imgur.com/BDcIbxQ.png",
"https://i.imgur.com/T9j98fL.png",
"https://i.imgur.com/Wiob6Hw.png",
"https://i.imgur.com/IVLXNOp.png",
"https://i.imgur.com/cy2GSCU.png",
"https://i.imgur.com/QaLm5z5.png",
"https://i.imgur.com/YO1lb4a.png",
"https://i.imgur.com/RQ4gY5c.png",
"https://i.imgur.com/8qzi7aS.png",
"https://i.imgur.com/W482EEm.png",
"https://i.imgur.com/mhsSsdN.png",
"https://i.imgur.com/iu8NWXP.png",
"https://i.imgur.com/azBwBak.png",
"https://i.imgur.com/VgSZiLK.png",
"https://i.imgur.com/7gLNSWl.png",
"https://i.imgur.com/gildukh.png",
"https://i.imgur.com/E1eryHf.png",
"https://i.imgur.com/j5RwW6J.png",
"https://i.imgur.com/AaiykNT.png",
"https://i.imgur.com/UksA18U.png",
"https://i.imgur.com/TRo3QAl.png",
"https://i.imgur.com/8XaYWQ0.png",
"https://i.imgur.com/2R7Je5Y.png",
"https://i.imgur.com/UD6O1Wq.png",
"https://i.imgur.com/clGVEZi.png",
"https://i.imgur.com/z32x65u.png",
"https://i.imgur.com/1VsLAok.png",
"https://i.imgur.com/C7DAxvo.png",
"https://i.imgur.com/XbGquZB.png",
"https://i.imgur.com/5mHYS2Z.png",
"https://i.imgur.com/OEf8cK5.png",
"https://i.imgur.com/7huMhzJ.png",
"https://i.imgur.com/JHY7DPR.png",
"https://i.imgur.com/2s4sgZh.png",
"https://i.imgur.com/oX3NXnN.png",
"https://i.imgur.com/u2FPd8N.png",
"https://i.imgur.com/3IEStfz.png",
"https://i.imgur.com/XOw5fgW.png",
"https://i.imgur.com/Ml5PeVs.png",
"https://i.imgur.com/isVYFVv.png",
"https://i.imgur.com/AD2ibRG.png",
"https://i.imgur.com/S6uoEcr.png",
"https://i.imgur.com/7eIN3vS.png",
"https://i.imgur.com/XHqcJlN.png",
"https://i.imgur.com/rjQNqM1.png",
"https://i.imgur.com/w5DbjyH.png",
"https://i.imgur.com/hccRRER.png",
"https://i.imgur.com/K1LJYpn.png",
"https://i.imgur.com/ajG1Jkr.png",
"https://i.imgur.com/mTXgZDR.png",
"https://i.imgur.com/qzjhuee.png",
"https://i.imgur.com/kk7LySa.png",
"https://i.imgur.com/nrM8YqE.png",
"https://i.imgur.com/2lAmIuB.png",
"https://i.imgur.com/xCS1eCG.png",
"https://i.imgur.com/qLsgjim.png",
"https://i.imgur.com/djFHKAQ.png",
"https://i.imgur.com/9Q4UQYL.png",
"https://i.imgur.com/U9SwO4w.png",
"https://i.imgur.com/ZhEqTD1.png",
"https://i.imgur.com/aQPUJNc.png",
"https://i.imgur.com/Fst76yn.png",
"https://i.imgur.com/mD3uvrM.png",
"https://i.imgur.com/fpxtuO5.png",

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
save_path = "./Dreambooth-Stable-Diffusion/training_images"
if not os.path.exists(save_path):
 os.mkdir(save_path)
[image.save(f"{save_path}/{i}.png", format="png") for i, image in enumerate(images)]
image_grid(images, 1, len(images))


# In[ ]:



