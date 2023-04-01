import os
import cv2
import numpy as np


def imshow(img):
    cv2.imshow('img', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def manual_kmeans(img):
    img_convert = img.reshape((-1, 3))
    img_convert = np.float32(img_convert)

    k = 2  # need 2 regions : back and foreground
    attempts = 25
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.1)

    _, labels, centers = cv2.kmeans(img_convert, k, None, criteria, attempts, cv2.KMEANS_RANDOM_CENTERS)

    centers = np.uint8(centers)
    res = centers[labels.flatten()]

    result_image = res.reshape(img.shape)

    gray_res = cv2.cvtColor(result_image, cv2.COLOR_BGR2GRAY)
    _, res = cv2.threshold(gray_res, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    return res


directory = "black/"

for filename in os.listdir(directory):
    og = cv2.imread(directory + filename)
    img = cv2.GaussianBlur(og, (5, 5), 0)

    segmentation = manual_kmeans(img)

    contours, hierarchy = cv2.findContours(segmentation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) > 0:
        cnt = contours[0]
        pokemon = cv2.drawContours(segmentation, [cnt], 0, (255, 255, 255), thickness=cv2.FILLED)
        pokemon = cv2.cvtColor(pokemon, cv2.COLOR_GRAY2BGR)

        img2 = og.copy()
        og = cv2.bitwise_not(og, img2)
        cc = cv2.bitwise_xor(og, pokemon)
        cv2.imwrite("res/" + filename, cc)
