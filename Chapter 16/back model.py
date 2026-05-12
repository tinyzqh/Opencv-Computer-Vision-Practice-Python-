import numpy as np
import cv2

# Classic demo video
cap = cv2.VideoCapture('test.avi')
# Structuring element for morphological operations
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
# Mixture-of-Gaussians background subtractor
fgbg = cv2.createBackgroundSubtractorMOG2()

while(True):
    ret, frame = cap.read()
    fgmask = fgbg.apply(frame)
    # Opening removes small noise from the foreground mask
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
    # Find contours of foreground objects
    contours, hierarchy = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for c in contours:
        # Perimeter of the current contour
        perimeter = cv2.arcLength(c,True)
        if perimeter > 188:
            # Axis-aligned bounding box (no rotation)
            x,y,w,h = cv2.boundingRect(c)
            # Draw the box on the original frame
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)

    cv2.imshow('frame',frame)
    cv2.imshow('fgmask', fgmask)
    k = cv2.waitKey(150) & 0xff
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()
