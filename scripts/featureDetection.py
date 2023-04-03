import cv2
import argparse
import numpy as np


def sift_features(img, name):
    count = 0
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    sift = cv2.SIFT_create()
    kp, des = sift.detectAndCompute(gray_img, None)

    # kp_img = cv2.drawKeyPoints(img, kp, None, color=(0, 255, 0),flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    kp_img = img.copy()
    for marker in kp:
        kp_img = cv2.drawMarker(kp_img, tuple(int(i) for i in marker.pt), color=(0, 255, 0))
        count += 1

    print(name,count)
    cv2.imshow(name, kp_img)
    cv2.waitKey()


def orb_feature_detection(img1, name):
    count = 0
    gray_img = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

    orb = cv2.ORB_create(nfeatures=2000)
    kp, des = orb.detectAndCompute(gray_img, None)

    kp_img = img1.copy()
    for marker in kp:
        kp_img = cv2.drawMarker(kp_img, tuple(int(i) for i in marker.pt), color=(0, 255, 0))
        count += 1

    print(name, count)

    cv2.imshow(name, kp_img)
    cv2.waitKey()


def surf_features(img):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    surf = cv2.xfeatures2d.SURF_create()
    kp, des = surf.detectAndCompute(gray_img, None)

    count = 0

    kp_img = img.copy()
    for marker in kp:
        kp_img = cv2.drawMarker(kp_img, tuple(int(i) for i in marker.pt), color=(0, 255, 0))
        count += 1

    print(count)

    cv2.imshow("SURF", kp_img)
    cv2.waitKey()

def harris_corner(img, name):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    gray = np.float32(gray)
    dst = cv2.cornerHarris(gray,2,3,0.04)

    dst = cv2.dilate(dst, None)

    img[dst>0.01*dst.max()] = [0,0,255]

    cv2.imshow(name, img)
    cv2.waitKey()


img1 = cv2.imread("car_images/Coburg2_left/1.jpg")
img2 = cv2.imread("car_images/Coburg2_front/1.jpg")
sift_features(img1, "SIFT Left")
sift_features(img2, "SIFT Front")
orb_feature_detection(img1, "ORB Left")
orb_feature_detection(img2, "ORB Front")
#surf_features(img)
harris_corner(img1, "Corner_Harris Left")
harris_corner(img2, "Corner_Harris Front")
