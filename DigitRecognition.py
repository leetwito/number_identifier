import cv2
import numpy as np
import sklearn

img = cv2.imread('big-Num1.jpg', 0)
ret, thresh = cv2.threshold(img, 40, 255, cv2.THRESH_BINARY)  # ToDo - set threshold (40) automatically
thresh = abs(thresh.astype(float) - 255)
thresh = thresh.astype('uint8')
cv2.imshow("thresh", thresh)
cv2.waitKey(0)


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

rotated = (rotated > 127).astype('uint8') * 255
data_mat = np.zeros((len(np.unique(markers))-1, 28*28))
for i in np.unique(markers)[1:]:
    marker = (markers == i).astype('uint8') * 255
    _, contours, hierarchy = cv2.findContours(marker, 2, 2)
    cnt = contours[0]
    x, y, w, h = cv2.boundingRect(cnt)
    #cv2.rectangle(marker, (x, y), (x + w, y + h), (70, 255, 0))

    square_edge = np.maximum(h, w)
    square_edge = int(square_edge * 1.4)
    top_bottom_padding = ((square_edge-h)/2, (square_edge-h)/2)
    right_left_padding = ((square_edge-w)/2, (square_edge-w)/2)
    only_digit = marker[y:y+h, x:x+w]
    only_digit = np.pad(only_digit, [top_bottom_padding, right_left_padding], mode='constant')
    resized_image = cv2.resize(only_digit, (28, 28), interpolation = cv2.INTER_AREA)
    resized_image = (resized_image > 40).astype('uint8') * 255
    reshaped_image = np.reshape(resized_image, (1,28*28))
    data_mat[i-1,:] = reshaped_image.astype('int32')
    # resized_image = only_digit
    cv2.imshow("marker" + str(i), resized_image)
    cv2.waitKey(0)

# cv2.imshow(" kaka", data_mat)
# cv2.waitKey(0)
# print data_mat.shape, np.max(data_mat), np.min(data_mat)
# show the output image
# cv2.imshow("Rotated Segmentation", rotated)
#cv2.imshow("rotated", rotated)
# cv2.imshow("Input", thresh)
global data_mat