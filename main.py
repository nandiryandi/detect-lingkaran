import cv2
import numpy as np

image = cv2.imread('blobs.jpg')

#Parameters
params = cv2.SimpleBlobDetector_Params()
params.filterByArea = True
params.minArea = 100

#set circularity filtering parameters
params.filterByCircularity = True
params.minCircularity = 0.9

params.filterByConvexity = True
params.minConvexity = 0.2

params.filterByInertia = True
params.minInertiaRatio = 0.01

#buat detektor dengan parameter
detector = cv2.SimpleBlobDetector_create(params)

#medeteksi blobs
keypoints = detector.detect(image)

#red circles
blank = np.zeros((1,1))
blobs = cv2.drawKeypoints(image, keypoints, blank, (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

number_of_blobs = len(keypoints)
text = "Lingkaran yang di deteksi : " + str(len(keypoints))
cv2.putText(blobs, text, (20, 550), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 100, 255), 2)

#memunculkan gambar yang di deteksi
cv2.imshow('Summon Image', blobs)
cv2.waitKey(0)
cv2.destroyAllWindows()

