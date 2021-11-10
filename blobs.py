import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image
image = cv2.imread('photos/face.jpeg', 0)


## blobs

# Read image
img = cv2.imread("photos/blob.png", cv2.IMREAD_GRAYSCALE)

# Setup SimpleBlobDetector parameters.
params = cv2.SimpleBlobDetector_Params()

# Filter by Convexity
params.filterByConvexity = True
params.minConvexity = 0.0

# Filter by Circularity
params.filterByCircularity = True
params.minCircularity = 0.5

# Set up the detector with default parameters.
detector = cv2.SimpleBlobDetector_create(params)

# Detect blobs.
keypoints = detector.detect(img)

# Draw detected blobs as red circles.
# cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
img_with_keypoints = cv2.drawKeypoints(img, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

f4 = plt.figure("blob")
plt.imshow(img_with_keypoints,  cmap="gray", vmin=0, vmax=255)
plt.title("Edges of Picture")

plt.savefig('plots/blob-plot.png')


plt.show()

