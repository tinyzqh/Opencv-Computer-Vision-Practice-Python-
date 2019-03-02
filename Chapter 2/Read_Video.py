import cv2
vc = cv2.VideoCapture('chaplin.mp4')
# 判断是否能够读取视频
if vc.isOpened():
    open, frame = vc.read()
else:
    open =False
while open:
    ret, frame = vc.read()
    if frame is None:
        break
    if ret == True:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cv2.imshow('result', gray)
        if cv2.waitKey(100) & 0xFF == 27:
            break
vc.release()
cv2.destroyAllWindows()
