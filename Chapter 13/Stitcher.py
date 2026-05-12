import numpy as np
import cv2

class Stitcher:

    # Main stitching routine
    def stitch(self, images, ratio=0.75, reprojThresh=4.0,showMatches=False):
        # Unpack the input images
        (imageB, imageA) = images
        # Detect SIFT keypoints and compute descriptors for image A and B
        (kpsA, featuresA) = self.detectAndDescribe(imageA)
        (kpsB, featuresB) = self.detectAndDescribe(imageB)

        # Match every feature point between the two images
        M = self.matchKeypoints(kpsA, kpsB, featuresA, featuresB, ratio, reprojThresh)

        # If no matches were returned, abort
        if M is None:
            return None

        # Otherwise unpack the match result
        # H is the 3x3 perspective transform matrix
        (matches, H, status) = M
        # Warp image A using the homography; result holds the warped image
        result = cv2.warpPerspective(imageA, H, (imageA.shape[1] + imageB.shape[1], imageA.shape[0]))
        self.cv_show('result', result)
        # Place image B on the left side of result
        result[0:imageB.shape[0], 0:imageB.shape[1]] = imageB
        self.cv_show('result', result)
        # Optionally render the match visualization
        if showMatches:
            # Build the visualization image
            vis = self.drawMatches(imageA, imageB, kpsA, kpsB, matches, status)
            # Return both
            return (result, vis)

        # Return the stitched result
        return result
    def cv_show(self,name,img):
        cv2.imshow(name, img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def detectAndDescribe(self, image):
        # Convert the color image to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Create the SIFT feature extractor
        descriptor = cv2.xfeatures2d.SIFT_create()
        # Detect SIFT keypoints and compute descriptors
        (kps, features) = descriptor.detectAndCompute(image, None)

        # Convert keypoints to a NumPy array
        kps = np.float32([kp.pt for kp in kps])

        # Return the keypoints and their feature descriptors
        return (kps, features)

    def matchKeypoints(self, kpsA, kpsB, featuresA, featuresB, ratio, reprojThresh):
        # Brute-force matcher
        matcher = cv2.BFMatcher()

        # KNN match SIFT features between A and B with K=2
        rawMatches = matcher.knnMatch(featuresA, featuresB, 2)

        matches = []
        for m in rawMatches:
            # Keep matches whose nearest / second-nearest distance ratio is below `ratio`
            if len(m) == 2 and m[0].distance < m[1].distance * ratio:
            # Store the index pair inside featuresA and featuresB
                matches.append((m[0].trainIdx, m[0].queryIdx))

        # Compute the homography once we have at least 4 reliable matches
        if len(matches) > 4:
            # Gather the matched point coordinates
            ptsA = np.float32([kpsA[i] for (_, i) in matches])
            ptsB = np.float32([kpsB[i] for (i, _) in matches])

            # Compute the perspective transform matrix
            (H, status) = cv2.findHomography(ptsA, ptsB, cv2.RANSAC, reprojThresh)

            # Return everything
            return (matches, H, status)

        # Not enough matches
        return None

    def drawMatches(self, imageA, imageB, kpsA, kpsB, matches, status):
        # Initialize the visualization by laying A and B side by side
        (hA, wA) = imageA.shape[:2]
        (hB, wB) = imageB.shape[:2]
        vis = np.zeros((max(hA, hB), wA + wB, 3), dtype="uint8")
        vis[0:hA, 0:wA] = imageA
        vis[0:hB, wA:] = imageB

        # Walk both lists together and draw the inlier matches
        for ((trainIdx, queryIdx), s) in zip(matches, status):
            # Only draw matches flagged as successful
            if s == 1:
                # Draw the match line
                ptA = (int(kpsA[queryIdx][0]), int(kpsA[queryIdx][1]))
                ptB = (int(kpsB[trainIdx][0]) + wA, int(kpsB[trainIdx][1]))
                cv2.line(vis, ptA, ptB, (0, 255, 0), 1)

        # Return the visualization
        return vis
