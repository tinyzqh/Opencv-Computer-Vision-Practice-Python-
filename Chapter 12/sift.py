import cv2
import numpy as np
img = cv2.imread('test_1.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
sift = cv2.xfeatures2d.SIFT_create()
kp = sift.detect(gray, None) #通常传进来一张灰度图，kp(KeyPoint)关键点
img = cv2.drawKeypoints(gray, kp, img)
cv2.imshow('drawKeypoints', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
kp, des = sift.compute(gray, kp)
print (np.array(kp).shape)
print(des.shape)
print(des[0])