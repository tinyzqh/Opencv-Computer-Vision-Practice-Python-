#导入工具包
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
# 参数
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

# SSD标签
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
	"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
	"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
	"sofa", "train", "tvmonitor"]

# 读取网络模型
print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

# 初始化
print("[INFO] starting video stream...")
vs = cv2.VideoCapture(args["video"])
writer = None

# 一会要追踪多个目标
trackers = []
labels = []

# 计算FPS
fps = FPS().start()

while True:
	# 读取一帧
	(grabbed, frame) = vs.read()

	# 是否是最后了
	if frame is None:
		break

	# 预处理操作
	(h, w) = frame.shape[:2]
	width=600
	r = width / float(w)
	dim = (width, int(h * r))
	frame = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)
	rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

	# 如果要将结果保存的话
	if args["output"] is not None and writer is None:
		fourcc = cv2.VideoWriter_fourcc(*"MJPG")
		writer = cv2.VideoWriter(args["output"], fourcc, 30,
			(frame.shape[1], frame.shape[0]), True)

	# 先检测 再追踪
	if len(trackers) == 0:
		# 获取blob数据
		(h, w) = frame.shape[:2]
		blob = cv2.dnn.blobFromImage(frame, 0.007843, (w, h), 127.5)

		# 得到检测结果
		net.setInput(blob)
		detections = net.forward()

		# 遍历得到的检测结果
		for i in np.arange(0, detections.shape[2]):
			# 能检测到多个结果，只保留概率高的
			confidence = detections[0, 0, i, 2]

			# 过滤
			if confidence > args["confidence"]:
				# extract the index of the class label from the
				# detections list
				idx = int(detections[0, 0, i, 1])
				label = CLASSES[idx]

				# 只保留人的
				if CLASSES[idx] != "person":
					continue

				# 得到BBOX
				#print (detections[0, 0, i, 3:7])
				box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
				(startX, startY, endX, endY) = box.astype("int")

				# 使用dlib来进行目标追踪
				#http://dlib.net/python/index.html#dlib.correlation_tracker
				t = dlib.correlation_tracker()
				rect = dlib.rectangle(int(startX), int(startY), int(endX), int(endY))
				t.start_track(rgb, rect)

				# 保存结果
				labels.append(label)
				trackers.append(t)

				# 绘图
				cv2.rectangle(frame, (startX, startY), (endX, endY),
					(0, 255, 0), 2)
				cv2.putText(frame, label, (startX, startY - 15),
					cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)

	# 如果已经有了框，就可以直接追踪了
	else:
		# 每一个追踪器都要进行更新
		for (t, l) in zip(trackers, labels):
			t.update(rgb)
			pos = t.get_position()

			# 得到位置
			startX = int(pos.left())
			startY = int(pos.top())
			endX = int(pos.right())
			endY = int(pos.bottom())

			# 画出来
			cv2.rectangle(frame, (startX, startY), (endX, endY),
				(0, 255, 0), 2)
			cv2.putText(frame, l, (startX, startY - 15),
				cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)

	# 也可以把结果保存下来
	if writer is not None:
		writer.write(frame)

	# 显示
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF

	# 退出
	if key == 27:
		break

	# 计算FPS
	fps.update()


fps.stop()
print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

if writer is not None:
	writer.release()

cv2.destroyAllWindows()
vs.release()