from PIL import Image
import os
from resizeimage import resizeimage

# Set working directory
os.chdir('/path/to/testset')

image_name = raw_input('Enter image name: ')

with open(image_name, 'r+b') as f:
    with Image.open(f) as image:
        cover = resizeimage.resize_cover(image, [250, 250])
        cover.save(image_name, image.format)