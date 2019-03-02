import cv2
img_cat = cv2.imread('cat.jpg')
img_dog = cv2.imread('dog.jpg')
print('img_cat shape', img_cat.shape)
print('img_dog shape', img_dog.shape)
cv2.imshow('origin cat', img_cat)
img_cat_add = img_cat + 10
cv2.imshow('cat add 10', img_cat_add)
print('img_cat 0-5', img_cat[:5, :, 0])
print('img_cat_add 0-5', img_cat_add[:5, :, 0])
print('img_dog 0-5', img_dog[:5, :, 0])
cv2.waitKey(0)
cv2.destroyAllWindows()