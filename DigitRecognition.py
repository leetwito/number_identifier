import cv2
import numpy as np
import sklearn

img = cv2.imread('big-Num1.jpg', 0)
ret, thresh = cv2.threshold(img, 40, 255, cv2.THRESH_BINARY)  # ToDo - set threshold (40) automatically
print type(thresh[0, 0])
thresh = abs(thresh.astype(float) - 255)
thresh = thresh.astype('uint8')

# grab the (x, y) coordinates of all pixel values that
# are greater than zero, then use these coordinates to
# compute a rotated bounding box that contains all
# coordinates
coords = np.column_stack(np.where(thresh > 0))
angle = cv2.minAreaRect(coords)[-1]
# the `cv2.minAreaRect` function returns values in the
# range [-90, 0); as the rectangle rotates clockwise the
# returned angle trends to 0 -- in this special case we
# need to add 90 degrees to the angle
if angle < -45:
    #angle = -(90 + angle)
    angle = -angle

# otherwise, just take the inverse of the angle to make
# it positive
else:
    angle = - (90 + angle)

# *** ROTATION ***
(h, w) = thresh.shape[:2]
center = (w // 2, h // 2)
M = cv2.getRotationMatrix2D(center, angle, 1.0)
rotated = cv2.warpAffine(thresh, M, (w, h),
 	flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
print("[INFO] angle: {:.3f}".format(angle))
rotated = (rotated > 127).astype('uint8') * 255

# *** SEGMENTATION ***

ret, markers = cv2.connectedComponents(rotated, connectivity=8)
print markers
print np.max(markers)
print markers.shape
print np.unique(rotated)
print type(markers[0, 0])

rotated = (rotated > 127).astype('uint8') * 255

for i in np.unique(markers)[1:]:
    marker = (markers == i).astype('uint8') * 255
    cv2.imshow("marker" + str(i), marker)
    cv2.waitKey(0)

# show the output image
# cv2.imshow("Rotated Segmentation", rotated)
cv2.imshow("rotated", rotated)
# cv2.imshow("Input", thresh)
