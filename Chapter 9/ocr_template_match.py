# Import packages
from imutils import contours
import numpy as np
import argparse
import cv2
import myutils

# Parse arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to input image")
ap.add_argument("-t", "--template", required=True,
	help="path to template OCR-A image")
args = vars(ap.parse_args())

# Credit-card type lookup based on the first digit
FIRST_NUMBER = {
	"3": "American Express",
	"4": "Visa",
	"5": "MasterCard",
	"6": "Discover Card"
}
# Helper to show an image and wait for a key
def cv_show(name,img):
	cv2.imshow(name, img)
	cv2.waitKey(0)
	cv2.destroyAllWindows()
# Read the template image
img = cv2.imread(args["template"])
cv_show('img',img)
# Convert to grayscale
ref = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv_show('ref',ref)
# Convert to a binary image
ref = cv2.threshold(ref, 10, 255, cv2.THRESH_BINARY_INV)[1]
cv_show('ref',ref)

# Find contours
# cv2.findContours() expects a binary image (not grayscale).
# cv2.RETR_EXTERNAL keeps only the outer contours; cv2.CHAIN_APPROX_SIMPLE keeps only end points.
# Each element of the returned list is one contour in the image.

refCnts, hierarchy = cv2.findContours(ref.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

cv2.drawContours(img,refCnts,-1,(0,0,255),3)
cv_show('img',img)
print(len(refCnts))
refCnts = myutils.sort_contours(refCnts, method="left-to-right")[0]  # Sort left-to-right, top-to-bottom
digits = {}

# Iterate over every contour
for (i, c) in enumerate(refCnts):
	# Compute bounding rectangle and resize to a fixed size
	(x, y, w, h) = cv2.boundingRect(c)
	roi = ref[y:y + h, x:x + w]
	roi = cv2.resize(roi, (57, 88))

	# Map each digit to its template
	digits[i] = roi

# Initialize kernels
rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 3))
sqKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

# Read and preprocess the input image
image = cv2.imread(args["image"])
cv_show('image',image)
image = myutils.resize(image, width=300)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv_show('gray',gray)

# Top-hat operation to highlight bright regions
tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, rectKernel)
cv_show('tophat',tophat)
#
gradX = cv2.Sobel(tophat, ddepth=cv2.CV_32F, dx=1, dy=0,  # ksize=-1 is equivalent to a 3x3 kernel
	ksize=-1)

gradX = np.absolute(gradX)
(minVal, maxVal) = (np.min(gradX), np.max(gradX))
gradX = (255 * ((gradX - minVal) / (maxVal - minVal)))
gradX = gradX.astype("uint8")

print (np.array(gradX).shape)
cv_show('gradX',gradX)

# Closing operation (dilate then erode) to join digits together
gradX = cv2.morphologyEx(gradX, cv2.MORPH_CLOSE, rectKernel)
cv_show('gradX',gradX)
# THRESH_OTSU finds the optimal threshold automatically (ideal for bimodal images); set the threshold parameter to 0
thresh = cv2.threshold(gradX, 0, 255,
	cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
cv_show('thresh',thresh)

# Another closing operation
thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, sqKernel)  # one more closing
cv_show('thresh',thresh)

# Find contours
threshCnts, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
	cv2.CHAIN_APPROX_SIMPLE)
cnts = threshCnts
cur_img = image.copy()
cv2.drawContours(cur_img,cnts,-1,(0,0,255),3)
cv_show('img',cur_img)
locs = []

# Iterate over contours
for (i, c) in enumerate(cnts):
	# Compute the bounding rectangle
	(x, y, w, h) = cv2.boundingRect(c)
	ar = w / float(h)

	# Filter for regions of the right shape — credit-card groups are 4 digits each here
	if ar > 2.5 and ar < 4.0:

		if (w > 40 and w < 55) and (h > 10 and h < 20):
			# Keep regions that match
			locs.append((x, y, w, h))

# Sort the matched contours left to right
locs = sorted(locs, key=lambda x:x[0])
output = []

# Iterate through digits inside every group
for (i, (gX, gY, gW, gH)) in enumerate(locs):
	# initialize the list of group digits
	groupOutput = []

	# Extract each group by its coordinates
	group = gray[gY - 5:gY + gH + 5, gX - 5:gX + gW + 5]
	cv_show('group',group)
	# Preprocess
	group = cv2.threshold(group, 0, 255,
		cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
	cv_show('group',group)
	# Find contours for each group
	digitCnts, hierarchy = cv2.findContours(group.copy(), cv2.RETR_EXTERNAL,
		cv2.CHAIN_APPROX_SIMPLE)
	digitCnts = contours.sort_contours(digitCnts,
		method="left-to-right")[0]

	# Recognize each digit inside the group
	for c in digitCnts:
		# Find the contour of the current digit and resize it
		(x, y, w, h) = cv2.boundingRect(c)
		roi = group[y:y + h, x:x + w]
		roi = cv2.resize(roi, (57, 88))
		cv_show('roi',roi)

		# Matching scores
		scores = []

		# Score against every template
		for (digit, digitROI) in digits.items():
			# Template matching
			result = cv2.matchTemplate(roi, digitROI,
				cv2.TM_CCOEFF)
			(_, score, _, _) = cv2.minMaxLoc(result)
			scores.append(score)

		# Pick the best matching digit
		groupOutput.append(str(np.argmax(scores)))

	# Draw the result
	cv2.rectangle(image, (gX - 5, gY - 5),
		(gX + gW + 5, gY + gH + 5), (0, 0, 255), 1)
	cv2.putText(image, "".join(groupOutput), (gX, gY - 15),
		cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)

	# Accumulate the result
	output.extend(groupOutput)

# Print the result
print("Credit Card Type: {}".format(FIRST_NUMBER[output[0]]))
print("Credit Card #: {}".format("".join(output)))
cv2.imshow("Image", image)
cv2.waitKey(0)
