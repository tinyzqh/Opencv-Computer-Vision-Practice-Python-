import cv2
import numpy as np
img = cv2.imread('dige.png')
kernel = np.ones((3,3), np.uint8)
dilate = cv2.dilate(img, kernel, iterations=1)
cv2.imshow('img', img)
cv2.imshow('erosion', dilate)
cv2.waitKey(0)
cv2.destroyAllWindows()