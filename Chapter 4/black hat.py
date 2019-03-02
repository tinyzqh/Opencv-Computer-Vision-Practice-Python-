import cv2
import numpy as np
img = cv2.imread('dige.png')
kernel = np.ones((7,7),np.uint8)
blackhat  = cv2.morphologyEx(img,cv2.MORPH_BLACKHAT, kernel)
cv2.imshow('blackhat', blackhat)
cv2.waitKey(0)
cv2.destroyAllWindows()