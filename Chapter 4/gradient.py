import cv2
import numpy as np
pie = cv2.imread('pie.png')
kernel = np.ones((2,2),np.uint8)
gradient = cv2.morphologyEx(pie, cv2.MORPH_GRADIENT, kernel)
cv2.imshow('gradient', gradient)
cv2.waitKey(0)
cv2.destroyAllWindows()