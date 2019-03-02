# Region Of Interest
import cv2
img = cv2.imread('cat.jpg')
img2 = img[50:200, 100:400] # 切片读取感兴趣的区域
cv2.imshow('cat',img2)
cv2.waitKey(0)
cv2.destroyAllWindows()
