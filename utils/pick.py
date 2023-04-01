import cv2
import os
import shutil
import numpy as np

directory = "colorall/"

for filename in os.listdir(directory):
    im = cv2.imread(directory + filename)
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

    if np.mean(im) == 255:
        os.remove(directory + filename)
    elif 220 <= gray[0, 0]:
        shutil.copy(directory + filename, "res/")
