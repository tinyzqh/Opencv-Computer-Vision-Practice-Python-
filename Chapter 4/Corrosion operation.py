import cv2
import numpy as np
img = cv2.imread('dige.png')
cv2.imshow('dige', img)
kernel = np.ones((3, 5), np.uint8)
erosion = cv2.erode(img,kernel,iterations = 1)
cv2.imshow('corrosion',erosion)
cv2.waitKey(0)
cv2.destroyAllWindows()