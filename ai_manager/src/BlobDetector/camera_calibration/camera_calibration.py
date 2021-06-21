#!/usr/bin/env python
# Work in progress: not saved on pycharm/Git
import cv2
import numpy as np
import os
import glob
import yaml

# File only for camera_calibration, only need to do it once before using the rest of the programn.

# workingdir="/home/pi/Desktop/Captures/"
savedir = './Camera_data/'  # !!

# Defining the dimensions of checkerboard
CHECKERBOARD = (8, 6)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 25, 0.01)

# Creating vector to store vectors of 3D points for each checkerboard image
objpoints = []
# Creating vector to store vectors of 2D points for each checkerboard image
imgpoints = []

# Defining the world coordinates for 3D points
objp = np.zeros((1, CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[0, :, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
prev_img_shape = None

# Extracting path of individual image stored in a given directory
images = glob.glob(
    './loin/Essai5/*.jpg')

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Find the chess board corners
    # If desired number of corners are found in the image then ret = true
    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD,
                                             cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)

    """
    If desired number of corner are detected,
    we refine the pixel coordinates and display 
    them on the images of checker board
    """
    if ret == True:
        objpoints.append(objp)
        # refining pixel coordinates for given 2d points.
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

        imgpoints.append(corners2)
        # Draw and display the corners
        img = cv2.drawChessboardCorners(img, CHECKERBOARD, corners2, ret)

    cv2.imshow('img', img)
    cv2.waitKey(1000)

cv2.destroyAllWindows()

h, w = img.shape[:2]

"""
Performing camera calibration by 
passing the value of known 3D points (objpoints)
and corresponding pixel coordinates of the 
detected corners (imgpoints)
"""
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

# Printing  and saving the results

print("Camera matrix : \n")
print(mtx)
np.save(savedir + 'cam_mtx.npy', mtx)

print("Dist : \n")
print(dist)
np.save(savedir + 'dist.npy', dist)

### UNDISTORSION ####

# Refining the camera matrix using parameters obtained by calibration
new_camera_mtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))

print("Region of Interest: \n")
print(roi)
np.save(savedir + 'roi.npy', roi)

print("New Camera Matrix: \n")
# print(newcam_mtx)
np.save(savedir + 'newcam_mtx.npy', new_camera_mtx)
print(np.load(savedir + 'newcam_mtx.npy'))

inverse_newcam_mtx = np.linalg.inv(new_camera_mtx)
print("Inverse New Camera Matrix: \n")
print(inverse_newcam_mtx)
np.save(savedir + 'inverse_newcam_mtx.npy', mtx)
np.save(savedir + 'mtx.npy', inverse_newcam_mtx)
np.save(savedir + 'new_camera_mtx.npy', new_camera_mtx)

# Method 1 to undistort the image
dst = cv2.undistort(img, mtx, dist, None, new_camera_mtx)

# Displaying the undistorted image
cv2.imshow("undistorted image", dst)
cv2.waitKey(0)

# Closing all remaining windows
cv2.destroyAllWindows()
