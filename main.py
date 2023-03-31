# -*- coding: utf-8 -*-
"""pokegan.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1bkmkTJFV0mUUC6tmP-9UBU0tBopsUML3


from google.colab import drive
drive.mount("/content/gdrive", force_remount=True)

!pip install kaggle

import os
os.environ['KAGGLE_CONFIG_DIR'] = "/content/gdrive/MyDrive/Kaggle/"

!kaggle datasets download -d thomassirvent/all-pokemon -p /content/sample_data/data/ --unzip
"""

import random
import torch
from data_utils import prepare_data
from gan_models import get_generator, get_discriminator, train
from visualize import plot_images, plot_losses, save_gif, side_by_side

manualSeed = 999
# manualSeed = random.randint(1, 10000) # use if you want new results
random.seed(manualSeed)
torch.manual_seed(manualSeed)


if __name__ == '__main__':

    nb_gpu = 1

    # Decide which device we want to run on
    device = torch.device("cuda:0" if (torch.cuda.is_available() and nb_gpu > 0) else "cpu")
    print("Running on", device)

    dataset = prepare_data(dataroot="data/all_poke/", image_size=64, batch_size=32, workers=2)
    plot_images(dataset, device)

    discriminator, optiD = get_discriminator(device, nb_gpu)
    generator, optiG = get_generator(device, nb_gpu)

    img_list, D_losses, G_losses = train(dataset, discriminator, optiD, generator, optiG, device, num_epochs=2)

    plot_losses(D_losses, G_losses)

    save_gif(img_list)

    side_by_side(dataset, device, img_list)

    print("End of program")
