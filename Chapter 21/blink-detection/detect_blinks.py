# Import packages
from scipy.spatial import distance as dist
from collections import OrderedDict
import numpy as np
import argparse
import time
import dlib
import cv2

FACIAL_LANDMARKS_68_IDXS = OrderedDict([
	("mouth", (48, 68)),
	("right_eyebrow", (17, 22)),
	("left_eyebrow", (22, 27)),
	("right_eye", (36, 42)),
	("left_eye", (42, 48)),
	("nose", (27, 36)),
	("jaw", (0, 17))
])

# http://vision.fe.uni-lj.si/cvww2016/proceedings/papers/05.pdf
def eye_aspect_ratio(eye):
	# Vertical distances
	A = dist.euclidean(eye[1], eye[5])
	B = dist.euclidean(eye[2], eye[4])
	# Horizontal distance
	C = dist.euclidean(eye[0], eye[3])
	# Eye Aspect Ratio (EAR)
	ear = (A + B) / (2.0 * C)
	return ear

# Parse arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", required=True,
	help="path to facial landmark predictor")
ap.add_argument("-v", "--video", type=str, default="",
	help="path to input video file")
args = vars(ap.parse_args())

# Decision thresholds
EYE_AR_THRESH = 0.3
EYE_AR_CONSEC_FRAMES = 3

# Initialize counters
COUNTER = 0
TOTAL = 0

# Detector and landmark predictor
print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])

# Grab the two eye-region indices
(lStart, lEnd) = FACIAL_LANDMARKS_68_IDXS["left_eye"]
(rStart, rEnd) = FACIAL_LANDMARKS_68_IDXS["right_eye"]

# Open the video stream
print("[INFO] starting video stream thread...")
vs = cv2.VideoCapture(args["video"])
#vs = FileVideoStream(args["video"]).start()
time.sleep(1.0)

def shape_to_np(shape, dtype="int"):
	# Allocate a 68x2 array
	coords = np.zeros((shape.num_parts, 2), dtype=dtype)
	# Walk through every landmark
	# Store its (x, y) coordinates
	for i in range(0, shape.num_parts):
		coords[i] = (shape.part(i).x, shape.part(i).y)
	return coords

# Iterate over every frame
while True:
	# Preprocessing
	frame = vs.read()[1]
	if frame is None:
		break

	(h, w) = frame.shape[:2]
	width=1200
	r = width / float(w)
	dim = (width, int(h * r))
	frame = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	# Detect faces
	rects = detector(gray, 0)

	# Iterate over every detected face
	for rect in rects:
		# Locate landmarks
		shape = predictor(gray, rect)
		shape = shape_to_np(shape)

		# Compute EAR for both eyes
		leftEye = shape[lStart:lEnd]
		rightEye = shape[rStart:rEnd]
		leftEAR = eye_aspect_ratio(leftEye)
		rightEAR = eye_aspect_ratio(rightEye)

		# Average them
		ear = (leftEAR + rightEAR) / 2.0

		# Draw the eye regions
		leftEyeHull = cv2.convexHull(leftEye)
		rightEyeHull = cv2.convexHull(rightEye)
		cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
		cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

		# Check the threshold
		if ear < EYE_AR_THRESH:
			COUNTER += 1

		else:
			# If eyes were closed for enough consecutive frames, register a blink
			if COUNTER >= EYE_AR_CONSEC_FRAMES:
				TOTAL += 1

			# Reset the counter
			COUNTER = 0

		# Display the result on the frame
		cv2.putText(frame, "Blinks: {}".format(TOTAL), (10, 30),
			cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
		cv2.putText(frame, "EAR: {:.2f}".format(ear), (300, 30),
			cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

	cv2.imshow("Frame", frame)
	key = cv2.waitKey(10) & 0xFF

	if key == 27:
		break

vs.release()
cv2.destroyAllWindows()
