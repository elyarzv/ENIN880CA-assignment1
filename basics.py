import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image
image = cv2.imread('photos/face.jpeg', 0)

## Basics
        
# Inverting the image
inverted = np.invert(image)

# applying threshhold
ret,thresh = cv2.threshold(image, 120, 255, cv2.THRESH_BINARY)

# applying edge detection
edge = cv2.Canny(image,80,200)

# saving the imaged in photo/ directory
cv2.imwrite('photos/inverted.jpeg', inverted)
cv2.imwrite('photos/thresh.jpeg', thresh)
cv2.imwrite('photos/edge.jpeg', edge)

# plotting the images resulted in basic  
f = plt.figure("basics")

f.add_subplot(3,2,1)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title("Original Picture")

f.add_subplot(3,2,2)
plt.imshow(inverted, cmap="gray", vmin=0, vmax=255)
plt.title("Negative Picture")

f.add_subplot(3,2,3)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title("Original Picture")

f.add_subplot(3,2,4)
plt.imshow(thresh, cmap="gray", vmin=0, vmax=255)
plt.title("Threshhold Picture")

f.add_subplot(3,2,5)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title("Original Picture")

f.add_subplot(3,2,6)
plt.imshow(edge, cmap="gray", vmin=0, vmax=255)
plt.title("Edge Detected Picture")

plt.savefig('plots/basics-plot.png')


plt.show()

