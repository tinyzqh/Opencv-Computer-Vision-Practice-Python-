import argparse
import time
import cv2
import numpy as np

# Parse arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", type=str,
	help="path to input video file")
ap.add_argument("-t", "--tracker", type=str, default="kcf",
	help="OpenCV object tracker type")
args = vars(ap.parse_args())

# Trackers shipped with OpenCV (legacy namespace since OpenCV 4.5)
OPENCV_OBJECT_TRACKERS = {
	"csrt": cv2.legacy.TrackerCSRT_create,
	"kcf": cv2.legacy.TrackerKCF_create,
	"boosting": cv2.legacy.TrackerBoosting_create,
	"mil": cv2.legacy.TrackerMIL_create,
	"tld": cv2.legacy.TrackerTLD_create,
	"medianflow": cv2.legacy.TrackerMedianFlow_create,
	"mosse": cv2.legacy.TrackerMOSSE_create
}

# Instantiate OpenCV's multi-object tracker
trackers = cv2.legacy.MultiTracker_create()
vs = cv2.VideoCapture(args["video"])

# Video loop
while True:
	# Grab the current frame
	frame = vs.read()
	# (success, data)
	frame = frame[1]
	# Stop when the video ends
	if frame is None:
		break

	# Resize every frame
	(h, w) = frame.shape[:2]
	width=600
	r = width / float(w)
	dim = (width, int(h * r))
	frame = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)

	# Tracking results
	(success, boxes) = trackers.update(frame)

	# Draw bounding boxes
	for box in boxes:
		(x, y, w, h) = [int(v) for v in box]
		cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

	# Show the frame
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(100) & 0xFF

	if key == ord("s"):
		# Press "s" to select a new region to track
		box = cv2.selectROI("Frame", frame, fromCenter=False,
			showCrosshair=True)

		# Create a fresh tracker for that ROI
		tracker = OPENCV_OBJECT_TRACKERS[args["tracker"]]()
		trackers.add(tracker, frame, box)

	# Quit
	elif key == 27:
		break
vs.release()
cv2.destroyAllWindows()
