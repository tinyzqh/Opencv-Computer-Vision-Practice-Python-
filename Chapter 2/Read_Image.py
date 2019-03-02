import cv2
import matplotlib.pyplot as plt
import numpy as np
img = cv2.imread('cat.jpg', cv2.IMREAD_GRAYSCALE) # opencv默认读取BGR格式
# 显示图像，可以创建多个窗口
cv2.imshow('Cat', img)
# 等待，0表示键盘任意键终止，如果为1000代表1000毫秒结束显示
cv2.waitKey(0)
cv2.destroyAllWindows()
print(img.shape)
cv2.imwrite('cat_gray.png', img)
print(366*550)
print(img.size) #查看像素点的个数