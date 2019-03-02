import cv2
import matplotlib.pyplot as plt
img = cv2.imread('cat.jpg')
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow('img_gray)', img_gray)
ret1, thresh1 = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY)
ret2, thresh2 = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY_INV)
ret3, thresh3 = cv2.threshold(img_gray, 127, 255, cv2.THRESH_TRUNC)
ret4, thresh4 = cv2.threshold(img_gray, 127, 255, cv2.THRESH_TOZERO)
ret5, thresh5 = cv2.threshold(img_gray, 127, 255, cv2.THRESH_TOZERO_INV)
titles = ['Original Image', 'BINARY', 'BINARY_INV', 'TRUNC', 'TOZERO', 'TOZERO_INV']
images = [img, thresh1, thresh2, thresh3, thresh4, thresh5]
for i in range(6):
    plt.subplot(2, 3, i + 1), plt.imshow(images[i], 'gray')
    plt.title(titles[i])
    plt.xticks([]), plt.yticks([])
plt.show()
cv2.waitKey(0)
cv2.destroyAllWindows()
