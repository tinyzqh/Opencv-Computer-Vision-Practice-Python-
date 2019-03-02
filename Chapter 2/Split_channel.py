import cv2
img = cv2.imread('cat1.jpg')
b, g, r = cv2.split(img)
img1 = cv2.merge((b, g, r))
cv2.imshow('cat_1', img1)
cop_img = img1.copy()
cop_img[:,:,0] = 0
cop_img[:,:,1] = 0
cv2.imshow('cat_only_r', cop_img)
cv2.waitKey(0)
cv2.destroyAllWindows()