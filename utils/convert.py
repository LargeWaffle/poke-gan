from PIL import Image
import os

directory = "data_ready/"
c = 1
for filename in os.listdir(directory):
    if filename.endswith(".JPG"):
        im = Image.open(directory + filename)
        name = 'img' + str(c) + '.png'
        rgb_im = im.convert('RGB')
        rgb_im.save(directory + name)
        os.remove(directory + filename)
        c += 1

print(c)
