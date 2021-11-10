import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image
image = cv2.imread('photos/face.jpeg', 0)


# Hough Transform

# Read image
img = cv2.imread("photos/highway.jpeg")

# Convert the image to gray-scale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Find the edges in the image using canny detector
edges = cv2.Canny(gray, 50, 200)

# Detect points that form a line
#lines = cv2.HoughLinesP(edges, 1, np.pi/180, max_slider=50, minLineLength=10, maxLineGap=250)

# Draw lines on the image
#for line in lines:
#    x1, y1, x2, y2 = line[0]
#    cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 3)
    
minLineLength = 10
maxLineGap = 50
#lines = cv2.HoughLinesP(edges, 1, np.pi/150, 200,        minLineLength,maxLineGap)
#lines = cv2.HoughLinesP(edges, 1, np.pi/180, 20, minLineLength=5, maxLineGap=100)
lines = cv2.HoughLines(edges,1,np.pi/180,230)
for line in lines:
    rho,theta = line[0]
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a*rho
    y0 = b*rho
    x1 = int(x0 + 1000*(-b))
    y1 = int(y0 + 1000*(a))
    x2 = int(x0 - 1000*(-b))
    y2 = int(y0 - 1000*(a))
    cv2.line(img,(x1,y1),(x2,y2),(0,0,255),2)

# Show result
f6 = plt.figure("hough")
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title("Edges of Picture")

plt.savefig('plots/hough-plot.png')

plt.show()

