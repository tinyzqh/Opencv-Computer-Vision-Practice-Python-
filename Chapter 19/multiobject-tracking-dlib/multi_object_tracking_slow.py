# Import packages
from utils import FPS
import numpy as np
import argparse
import dlib
import cv2
"""
--prototxt mobilenet_ssd/MobileNetSSD_deploy.prototxt
--model mobilenet_ssd/MobileNetSSD_deploy.caffemodel
--video race.mp4
"""
# Arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--prototxt", required=True,
	help="path to Caffe 'deploy' prototxt file")
ap.add_argument("-m", "--model", required=True,
	help="path to Caffe pre-trained model")
ap.add_argument("-v", "--video", required=True,
	help="path to input video file")
ap.add_argument("-o", "--output", type=str,
	help="path to optional output video file")
ap.add_argument("-c", "--confidence", type=float, default=0.2,
	help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

# SSD class labels
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
	"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
	"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
	"sofa", "train", "tvmonitor"]

# Load the network
print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

# Initialize the video stream
print("[INFO] starting video stream...")
vs = cv2.VideoCapture(args["video"])
writer = None

# We'll track multiple objects
trackers = []
labels = []

# FPS counter
fps = FPS().start()

while True:
	# Read a frame
	(grabbed, frame) = vs.read()

	# End of video
	if frame is None:
		break

	# Preprocessing
	(h, w) = frame.shape[:2]
	width=600
	r = width / float(w)
	dim = (width, int(h * r))
	frame = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)
	rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

	# Initialize the writer if we are saving results
	if args["output"] is not None and writer is None:
		fourcc = cv2.VideoWriter_fourcc(*"MJPG")
		writer = cv2.VideoWriter(args["output"], fourcc, 30,
			(frame.shape[1], frame.shape[0]), True)

	# Detect first, then track
	if len(trackers) == 0:
		# Build the input blob
		(h, w) = frame.shape[:2]
		blob = cv2.dnn.blobFromImage(frame, 0.007843, (w, h), 127.5)

		# Run detection
		net.setInput(blob)
		detections = net.forward()

		# Walk through every detection
		for i in np.arange(0, detections.shape[2]):
			# Many detections come back — only keep high-confidence ones
			confidence = detections[0, 0, i, 2]

			# Confidence filter
			if confidence > args["confidence"]:
				# extract the index of the class label from the
				# detections list
				idx = int(detections[0, 0, i, 1])
				label = CLASSES[idx]

				# Keep only the "person" class
				if CLASSES[idx] != "person":
					continue

				# Bounding box in pixel coordinates
				#print (detections[0, 0, i, 3:7])
				box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
				(startX, startY, endX, endY) = box.astype("int")

				# Track with dlib's correlation tracker
				# http://dlib.net/python/index.html#dlib.correlation_tracker
				t = dlib.correlation_tracker()
				rect = dlib.rectangle(int(startX), int(startY), int(endX), int(endY))
				t.start_track(rgb, rect)

				# Save results
				labels.append(label)
				trackers.append(t)

				# Draw the detection
				cv2.rectangle(frame, (startX, startY), (endX, endY),
					(0, 255, 0), 2)
				cv2.putText(frame, label, (startX, startY - 15),
					cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)

	# Otherwise we already have boxes — just keep tracking
	else:
		# Update every tracker
		for (t, l) in zip(trackers, labels):
			t.update(rgb)
			pos = t.get_position()

			# Read the new position
			startX = int(pos.left())
			startY = int(pos.top())
			endX = int(pos.right())
			endY = int(pos.bottom())

			# Draw it
			cv2.rectangle(frame, (startX, startY), (endX, endY),
				(0, 255, 0), 2)
			cv2.putText(frame, l, (startX, startY - 15),
				cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)

	# Save the frame if requested
	if writer is not None:
		writer.write(frame)

	# Show the frame
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF

	# Quit
	if key == 27:
		break

	# Update FPS counter
	fps.update()


fps.stop()
print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

if writer is not None:
	writer.release()

cv2.destroyAllWindows()
vs.release()
