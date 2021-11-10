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
f = plt.figure(0)

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


## Convolution

# creating kernel matrix
kernel = np.ones((21, 21), np.float32) / (21**2)

# convoluting the image withh kernel
img_uniform = cv2.filter2D(image, -1, kernel)

#applying Gaussian blur on image
img_gaussian = cv2.GaussianBlur(image, (21,21), 0)

# saving image in photos folder
cv2.imwrite('photos/uniform.jpeg', img_uniform)
cv2.imwrite('photos/guassian.jpeg', img_gaussian)

# show the image 
f2 = plt.figure(1)

f2.add_subplot(2,2,1)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title("Original Picture")

f2.add_subplot(2,2,2)
plt.imshow(cv2.cvtColor(img_uniform, cv2.COLOR_BGR2RGB))
plt.title("Blurred Picture with uniform kernel")

f2.add_subplot(2,2,3)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title("Original Picture")

f2.add_subplot(2,2,4)
plt.imshow(cv2.cvtColor(img_gaussian, cv2.COLOR_BGR2RGB))
plt.title("Blurred Picture with guassian kernel")


## Template Matching

# Read the main image
img_rgb = cv2.imread("photos/face.jpeg")
img = cv2.imread("photos/face.jpeg")
 
# Convert it to grayscale
img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
 
# Read the template
template = cv2.imread("photos/template.jpeg", 0)
 
# Store width and height of template in w and h
w, h = template.shape[::-1]
 
# Perform match operations.
res = cv2.matchTemplate(img_gray,template,cv2.TM_CCOEFF_NORMED)
 
# Specify a threshold
threshold = 0.8
 
# Store the coordinates of matched area in a numpy array
loc = np.where( res >= threshold)
 
# Draw a rectangle around the matched region.
for pt in zip(*loc[::-1]):
    cv2.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (0,255,255), 2)

# Show the final image with the matched area.
# Show the image 
f3 = plt.figure(0)

f3.add_subplot(1,2,1)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title("Original Picture")

f3.add_subplot(1,2,2)
plt.imshow(cv2.cvtColor(img_rgb, cv2.COLOR_BGR2RGB))
plt.title("Gaussian Blurred Picture")

plt.show()
