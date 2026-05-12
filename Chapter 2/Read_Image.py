import cv2
import matplotlib.pyplot as plt
import numpy as np
img = cv2.imread('cat.jpg', cv2.IMREAD_GRAYSCALE)  # OpenCV reads images in BGR format by default
# Display the image; multiple windows can be created
cv2.imshow('Cat', img)
# Wait for a key press: 0 means wait indefinitely, 1000 would wait 1000 ms
cv2.waitKey(0)
cv2.destroyAllWindows()
print(img.shape)
cv2.imwrite('cat_gray.png', img)
print(366*550)
print(img.size)  # total number of pixels