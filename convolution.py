import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image
image = cv2.imread('photos/face.jpeg', 0)

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
f2 = plt.figure("convolution")

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

plt.savefig('plots/convolution-plot.png')


plt.show()

