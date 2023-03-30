import cv2
import imutils as imutils
import numpy as np

# Pfade der Bilder
img_paths = ["stitching_images/stitching_im1.jpg", "stitching_images/stitching_im2.jpg", "stitching_images/stitching_im3.jpg"]

imgs = []

for i in range(len(img_paths)):
    imgs.append(cv2.imread(img_paths[i]))
    imgs[i] = cv2.resize(imgs[i],(0,0),fx=0.4, fy=0.4)


#cv2.imshow("1", imgs[0])
#cv2.imshow("2", imgs[1])
#cv2.imshow("3", imgs[2])

stitched_image = cv2.Stitcher.create()
(dummy, stitched_image) = stitched_image.stitch(imgs)

if dummy != cv2.STITCHER_OK:
    print("stitching unsuccessful...")
else:
    print("stitching successful")

cv2.imshow("final result", stitched_image)

cv2.waitKey(0)

stitched_image = cv2.copyMakeBorder(stitched_image, 10, 10, 10, 10, cv2.BORDER_CONSTANT, (0, 0, 0))

gray = cv2.cvtColor(stitched_image, cv2.COLOR_BGR2GRAY)
thresh_img = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)[1]

cv2.imshow("Threshold Image", thresh_img)
cv2.waitKey(0)

contours = cv2.findContours(thresh_img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

contours = imutils.grab_contours(contours)
areaOI = max(contours, key=cv2.contourArea)

mask = np.zeros(thresh_img.shape, dtype="uint8")
x, y, w, h = cv2.boundingRect(areaOI)
cv2.rectangle(mask, (x,y), (x + w, y + h), 255, -1)

minRectangle = mask.copy()
sub = mask.copy()

while cv2.countNonZero(sub) > 0:
    minRectangle = cv2.erode(minRectangle, None)
    sub = cv2.subtract(minRectangle, thresh_img)

contours = cv2.findContours(minRectangle.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

contours = imutils.grab_contours(contours)
areaOI = max(contours, key=cv2.contourArea)

cv2.imshow("minRectangle Image", minRectangle)
cv2.waitKey(0)

x, y, w, h = cv2.boundingRect(areaOI)

stitched_image = stitched_image[y:y + h, x:x + w]

cv2.imwrite("stitchedOutputProcessed.png", stitched_image)

cv2.imshow("Stitched Image Processed", stitched_image)

cv2.waitKey(0)