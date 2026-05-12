import numpy as np
import cv2

cap = cv2.VideoCapture('test.avi')

# Parameters for corner detection
feature_params = dict( maxCorners = 100,
                       qualityLevel = 0.3,
                       minDistance = 7)

# Lucas-Kanade parameters
lk_params = dict( winSize  = (15,15),
                  maxLevel = 2)

# Random color palette for trails
color = np.random.randint(0,255,(100,3))

# Grab the first frame
ret, old_frame = cap.read()
old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
# goodFeaturesToTrack returns the strongest corners. Inputs: image, max corners (efficiency cap),
# quality level (higher eigenvalues are better — used to filter), and min distance
# (any weaker corner within this radius of a stronger one is dropped).
p0 = cv2.goodFeaturesToTrack(old_gray, mask = None, **feature_params)

# Create an overlay mask to draw the trails on
mask = np.zeros_like(old_frame)

while(True):
    ret,frame = cap.read()
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Pass in the previous frame, the current frame, and the previous corner locations
    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

    # st == 1 marks points that were tracked successfully
    good_new = p1[st==1]
    good_old = p0[st==1]

    # Draw the motion trails
    for i,(new,old) in enumerate(zip(good_new,good_old)):
        a,b = new.ravel().astype(int)
        c,d = old.ravel().astype(int)
        mask = cv2.line(mask, (a,b),(c,d), color[i].tolist(), 2)
        frame = cv2.circle(frame,(a,b),5,color[i].tolist(),-1)
    img = cv2.add(frame,mask)

    cv2.imshow('frame',img)
    k = cv2.waitKey(150) & 0xff
    if k == 27:
        break

    # Update for the next iteration
    old_gray = frame_gray.copy()
    p0 = good_new.reshape(-1,1,2)

cv2.destroyAllWindows()
cap.release()
