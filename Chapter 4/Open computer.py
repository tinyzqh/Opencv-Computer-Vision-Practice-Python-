import numpy as np
import cv2
img = cv2.imread('dige.png')
kernel = np.ones((5,5), np.uint8)
opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
cv2.imshow('opening', opening)
cv2.waitKey(0)
cv2.destroyAllWindows()