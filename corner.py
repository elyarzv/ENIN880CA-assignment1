import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image
image = cv2.imread('photos/face.jpeg', 0)



## corner detection

# loading image
image = cv2.imread('photos/desk.jpg') 
  
# convert the input image into grayscale
img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 
  
# convert the data type 
img_gray = np.float32(img_gray) 
  
# implementing cv2.cornerHarris method 
hcd_img = cv2.cornerHarris(img_gray, 5, 5, 0.08) 
  
# marking dilated corners 
hcd_img = cv2.dilate(hcd_img, None) 
  
# reverting back to the original image
image[hcd_img > 0.01 * hcd_img.max()]=[0, 0, 255] 

#Show result
f7 = plt.figure("corner")
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title("Corners detected with Harris Algorithm")

plt.savefig('plots/corner-plot.png')


plt.show()

