import glob
import cv2
import os
import datetime

images = [cv2.imread(file) for file in glob.glob("/home/student1/ros_pictures/Camera_hautex256-1/fail/*.jpg")]

for frame in images:

    # taille du crop
    h = 224
    w = 224

    x = 0
    y = 0
    # réalisation du crop
    crop = frame[x:x + h, y:y + w]

    cv2.imwrite(os.path.join('/home/student1/ros_pictures/Working_Images/3000x224/fail',
                             'fail' + str(datetime.datetime.now()) + '.jpg'), crop)




