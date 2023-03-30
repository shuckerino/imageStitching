import cv2

# Pfade der Bilder
img_paths = ["pics/pic1.jpg", "pics/pic2.jpg"]

imgs = []

for i in range(len(img_paths)):
    imgs.append(cv2.imread(img_paths[i]))
    imgs[i] = cv2.resize(imgs[i],(0,0),fx=0.4, fy=0.4)


cv2.imshow("1", imgs[0])
cv2.imshow("2", imgs[1])

stitchy = cv2.Stitcher.create()
(dummy, output) = stitchy.stitch(imgs)

if dummy != cv2.STITCHER_OK:
    print("stitching ain't working man...")
else:
    print("yeah man, successful")

cv2.imshow("final result", output)

cv2.waitKey(0)