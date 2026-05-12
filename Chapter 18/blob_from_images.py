# Import packages
import utils_paths
import numpy as np
import cv2

# Load class labels
rows = open("synset_words.txt").read().strip().split("\n")
classes = [r[r.find(" ") + 1:].split(",")[0] for r in rows]

# Load the Caffe model
net = cv2.dnn.readNetFromCaffe("bvlc_googlenet.prototxt",
	"bvlc_googlenet.caffemodel")

# Image paths
imagePaths = sorted(list(utils_paths.list_images("images/")))

# Preprocess the first image
image = cv2.imread(imagePaths[0])
resized = cv2.resize(image, (224, 224))
# image, scalefactor, size, mean, swapRB
blob = cv2.dnn.blobFromImage(resized, 1, (224, 224), (104, 117, 123))
print("First Blob: {}".format(blob.shape))

# Run a forward pass
net.setInput(blob)
preds = net.forward()

# Sort and pick the top class
idx = np.argsort(preds[0])[::-1][0]
text = "Label: {}, {:.2f}%".format(classes[idx],
	preds[0][idx] * 100)
cv2.putText(image, text, (5, 25),  cv2.FONT_HERSHEY_SIMPLEX,
	0.7, (0, 0, 255), 2)

# Display the result
cv2.imshow("Image", image)
cv2.waitKey(0)

# Build a batch
images = []

# Same as above, but for a batch of images
for p in imagePaths[1:]:
	image = cv2.imread(p)
	image = cv2.resize(image, (224, 224))
	images.append(image)

# blobFromImages — note the plural "s"
blob = cv2.dnn.blobFromImages(images, 1, (224, 224), (104, 117, 123))
print("Second Blob: {}".format(blob.shape))

# Run inference on the batch
net.setInput(blob)
preds = net.forward()
for (i, p) in enumerate(imagePaths[1:]):
	image = cv2.imread(p)
	idx = np.argsort(preds[i])[::-1][0]
	text = "Label: {}, {:.2f}%".format(classes[idx],
		preds[i][idx] * 100)
	cv2.putText(image, text, (5, 25),  cv2.FONT_HERSHEY_SIMPLEX,
		0.7, (0, 0, 255), 2)
	cv2.imshow("Image", image)
	cv2.waitKey(0)
