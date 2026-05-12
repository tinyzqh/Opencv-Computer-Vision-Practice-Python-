# Import packages
import numpy as np
import argparse
import imutils
import cv2

# Parse arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to the input image")
args = vars(ap.parse_args())

# Ground-truth answer key
ANSWER_KEY = {0: 1, 1: 4, 2: 0, 3: 3, 4: 1}

def order_points(pts):
	# Four corner points in total
	rect = np.zeros((4, 2), dtype = "float32")

	# Indices 0-3 correspond to top-left, top-right, bottom-right, bottom-left
	# Top-left has the smallest x+y sum, bottom-right the largest
	s = pts.sum(axis = 1)
	rect[0] = pts[np.argmin(s)]
	rect[2] = pts[np.argmax(s)]

	# Top-right has the smallest y-x diff, bottom-left the largest
	diff = np.diff(pts, axis = 1)
	rect[1] = pts[np.argmin(diff)]
	rect[3] = pts[np.argmax(diff)]

	return rect

def four_point_transform(image, pts):
	# Order the input points
	rect = order_points(pts)
	(tl, tr, br, bl) = rect

	# Compute the width and height of the new image
	widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
	widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
	maxWidth = max(int(widthA), int(widthB))

	heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
	heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
	maxHeight = max(int(heightA), int(heightB))

	# Destination coordinates after the transform
	dst = np.array([
		[0, 0],
		[maxWidth - 1, 0],
		[maxWidth - 1, maxHeight - 1],
		[0, maxHeight - 1]], dtype = "float32")

	# Compute the transform matrix
	M = cv2.getPerspectiveTransform(rect, dst)
	warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

	# Return the warped image
	return warped
def sort_contours(cnts, method="left-to-right"):
    reverse = False
    i = 0
    if method == "right-to-left" or method == "bottom-to-top":
        reverse = True
    if method == "top-to-bottom" or method == "bottom-to-top":
        i = 1
    boundingBoxes = [cv2.boundingRect(c) for c in cnts]
    (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
                                        key=lambda b: b[1][i], reverse=reverse))
    return cnts, boundingBoxes
def cv_show(name,img):
        cv2.imshow(name, img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

# Preprocessing
image = cv2.imread(args["image"])
contours_img = image.copy()
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
cv_show('blurred',blurred)
edged = cv2.Canny(blurred, 75, 200)
cv_show('edged',edged)

# Contour detection
cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,
	cv2.CHAIN_APPROX_SIMPLE)[0]
cv2.drawContours(contours_img,cnts,-1,(0,0,255),3)
cv_show('contours_img',contours_img)
docCnt = None

# Ensure at least one contour was found
if len(cnts) > 0:
	# Sort contours by area, largest first
	cnts = sorted(cnts, key=cv2.contourArea, reverse=True)

	# Walk through every contour
	for c in cnts:
		# Approximate the contour
		peri = cv2.arcLength(c, True)
		approx = cv2.approxPolyDP(c, 0.02 * peri, True)

		# Ready to perform a perspective transform
		if len(approx) == 4:
			docCnt = approx
			break

# Apply the perspective transform

warped = four_point_transform(gray, docCnt.reshape(4, 2))
cv_show('warped',warped)
# Otsu's thresholding
thresh = cv2.threshold(warped, 0, 255,
	cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
cv_show('thresh',thresh)
thresh_Contours = thresh.copy()
# Find the contours of every bubble
cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
	cv2.CHAIN_APPROX_SIMPLE)[0]
cv2.drawContours(thresh_Contours,cnts,-1,(0,0,255),3)
cv_show('thresh_Contours',thresh_Contours)
questionCnts = []

# Walk through every contour
for c in cnts:
	# Compute aspect ratio and size
	(x, y, w, h) = cv2.boundingRect(c)
	ar = w / float(h)

	# Adjust the thresholds to match the actual document
	if w >= 20 and h >= 20 and ar >= 0.9 and ar <= 1.1:
		questionCnts.append(c)

# Sort bubbles top-to-bottom
questionCnts = sort_contours(questionCnts,
	method="top-to-bottom")[0]
correct = 0

# Each row has 5 bubbles
for (q, i) in enumerate(np.arange(0, len(questionCnts), 5)):
	# Sort within the row
	cnts = sort_contours(questionCnts[i:i + 5])[0]
	bubbled = None

	# Walk through every bubble
	for (j, c) in enumerate(cnts):
		# Build a mask to evaluate the bubble
		mask = np.zeros(thresh.shape, dtype="uint8")
		cv2.drawContours(mask, [c], -1, 255, -1)  # -1 means fill the contour
		cv_show('mask',mask)
		# Count non-zero pixels to decide whether the bubble is filled
		mask = cv2.bitwise_and(thresh, thresh, mask=mask)
		total = cv2.countNonZero(mask)

		# Track the bubble with the most filled pixels
		if bubbled is None or total > bubbled[0]:
			bubbled = (total, j)

	# Compare with the answer key
	color = (0, 0, 255)
	k = ANSWER_KEY[q]

	# Mark correct answers in green
	if k == bubbled[1]:
		color = (0, 255, 0)
		correct += 1

	# Draw the result
	cv2.drawContours(warped, [cnts[k]], -1, color, 3)


score = (correct / 5.0) * 100
print("[INFO] score: {:.2f}%".format(score))
cv2.putText(warped, "{:.2f}%".format(score), (10, 30),
	cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
cv2.imshow("Original", image)
cv2.imshow("Exam", warped)
cv2.waitKey(0)
