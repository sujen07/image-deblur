import os
import shutil
import random
from PIL import Image

data_path = 'blur-data'
targets = sorted(os.listdir(os.path.join(data_path, 'sharp')))
defocused_blurs = sorted(os.listdir(os.path.join(data_path, 'defocused_blurred')))
motion_blurs = sorted(os.listdir(os.path.join(data_path, 'motion_blurred')))

train_loc = 'data/train'
val_loc = 'data/validation'


os.makedirs(train_loc + '/lr', exist_ok=True)
os.makedirs(train_loc + '/hr', exist_ok=True)
os.makedirs(val_loc + '/lr', exist_ok=True)
os.makedirs(val_loc + '/hr', exist_ok=True)

counter = 0

def scale_image(image_path, scale_factor):
    with Image.open(image_path) as img:
        width, height = img.size
        new_size = (width // scale_factor, height // scale_factor)
        img_resized = img.resize(new_size, Image.LANCZOS)
        return img_resized

def save_scaled_image(image_path, scale_factor, save_path):
    scaled_image = scale_image(image_path, scale_factor)
    scaled_image.save(save_path)

for d, m, t in zip(defocused_blurs, motion_blurs, targets):
    d_path = os.path.join(data_path, 'defocused_blurred', d)
    m_path = os.path.join(data_path, 'motion_blurred', m)
    t_path = os.path.join(data_path, 'sharp', t)

    lr_loc = os.path.join(train_loc, 'lr')
    hr_loc = os.path.join(train_loc, 'hr')

    val_rand = random.randint(1, 100)
    if val_rand <= 10:
        lr_loc = os.path.join(val_loc, 'lr')
        hr_loc = os.path.join(val_loc, 'hr')

    # Define new filenames
    lr_filename = f"{counter}.jpg"
    hr_filename = f"{counter}.jpg"

    # Scale down the defocused blur image and save to lr folder
    save_scaled_image(d_path, 2, os.path.join(lr_loc, lr_filename))
    
    # Scale down the corresponding sharp image and save to hr folder with new filename
    save_scaled_image(t_path, 2, os.path.join(hr_loc, hr_filename))  # Adjust scale factor if needed

    # Increment the counter for the next set of images
    counter += 1

    # Define new filenames for motion blur
    lr_filename = f"{counter}.jpg"
    hr_filename = f"{counter}.jpg"

    # Scale down the motion blur image and save to lr folder
    #save_scaled_image(m_path, 5, os.path.join(lr_loc, lr_filename))
    
    # Scale down the corresponding sharp image and save to hr folder with new filename
    #save_scaled_image(t_path, 5, os.path.join(hr_loc, hr_filename))  # Adjust scale factor if needed

    # Increment the counter for the next set of images
    #counter += 1

print("Files copied, resized, and renamed successfully!")
"""

for lr, hr in zip(sorted(os.listdir('blurred_images')), sorted(os.listdir('hr'))):

    lr_loc = os.path.join(train_loc, 'lr')
    hr_loc = os.path.join(train_loc, 'hr')

    val_rand = random.randint(1, 100)
    if val_rand <= 10:
        lr_loc = os.path.join(val_loc, 'lr')
        hr_loc = os.path.join(val_loc, 'hr')

    if lr != hr:
        continue
    hr_img = Image.open(os.path.join('hr',hr))
    lr_img = Image.open(os.path.join('blurred_images',lr))

    hr_img = hr_img.resize(lr_img.size, Image.LANCZOS)

    hr_img.save(os.path.join(hr_loc, hr))
    lr_img.save(os.path.join(lr_loc, lr))
"""