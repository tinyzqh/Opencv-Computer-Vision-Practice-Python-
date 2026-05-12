from Stitcher import Stitcher
import cv2

# Read the images to stitch
imageA = cv2.imread("left_01.png")
imageB = cv2.imread("right_01.png")

# Stitch them into a panorama
stitcher = Stitcher()
(result, vis) = stitcher.stitch([imageA, imageB], showMatches=True)

# Show every image
cv2.imshow("Image A", imageA)
cv2.imshow("Image B", imageB)
cv2.imshow("Keypoint Matches", vis)
cv2.imshow("Result", result)
cv2.waitKey(0)
cv2.destroyAllWindows()
