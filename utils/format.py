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

    return result_image


img = cv2.imread('data/test.jpg')
img = cv2.GaussianBlur(img, (5, 5), 0)

res = manual_kmeans(img)

imshow(img)
imshow(res)
