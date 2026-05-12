# Region Of Interest
import cv2
img = cv2.imread('cat.jpg')
img2 = img[50:200, 100:400]  # Slice to extract the region of interest
cv2.imshow('cat',img2)
cv2.waitKey(0)
cv2.destroyAllWindows()
