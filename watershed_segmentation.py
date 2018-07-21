def histogram_equalise(gray):
	return cv2.equalizeHist(gray)


#code for Component Detection using Watershed Segmentation
import cv2
import numpy as np
from matplotlib import pyplot as plt
 
img = cv2.imread("D:\\work\\python codes\\opencv python\\images\\pcb images\\image7.jpg", 1)
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
equalised_gray = histogram_equalise(gray)
ret, thresh = cv2.threshold(equalised_gray ,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
# noise removal
kernel = np.ones((3, 3),np.uint8)
opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 9)
# sure background area
sure_bg = cv2.dilate(opening,kernel,iterations=1)
rows, cols = sure_bg.shape
for i in range(0, rows):
	for j in range(0, cols):
		sure_bg[i, j] = abs(sure_bg[i, j] - 255)


# Finding sure foreground area
dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,0)
ret, sure_fg = cv2.threshold(dist_transform,0.009*dist_transform.max(),255,0)
# Finding unknown region
sure_fg = np.uint8(sure_fg)
unknown = cv2.bitwise_not(cv2.bitwise_xor(sure_bg,sure_fg))


# Marker labelling
ret, markers = cv2.connectedComponents(sure_fg)
# Add one to all labels so that sure background is not 0, but 1
markers = markers+1
# Now, mark the region of unknown with zero
markers[unknown==255] = 0

markers = cv2.watershed(img,markers)
img[markers == -1] = [255,0,0]
plt.subplot(221)
plt.imshow(equalised_gray, cmap = "gray")
plt.title('Original Image')
plt.subplot(222)
plt.imshow(sure_fg, cmap = "gray")
plt.title('Sure Foreground')
plt.subplot(223)
plt.imshow(sure_bg, cmap = "gray")
plt.title('Sure Background')
plt.subplot(224)
plt.imshow(img)
plt.title('Detected Components')
plt.show()