import cv2
import numpy as np
from PerspectiveCalibration import PerspectiveCalibration

# global variables
image_path = './Calibration_allimages/webcam/circle/2021-05-04-164701.jpg'
image_coordinates = []

dPoint = PerspectiveCalibration()
dPoint.setup_camera()
def click_event(event, x, y, flags, params): 
      
    # checking for left mouse clicks 
    if event == cv2.EVENT_LBUTTONDOWN:


        image_coord = [x,y]
        xyz = dPoint.from_2d_to_3d(image_coord)


        print(xyz)
        # displaying the coordinates on the Shell 
        #print(x, ' ', y)
        images_coordinates = image_coordinates.append([x, y])
        # displaying the coordinates 
        # on the image window 
        font = cv2.FONT_HERSHEY_SIMPLEX 
        cv2.putText(img, str(int(xyz[0][0])) + ' , ' +
                    str(int(xyz[1][0])) + ' , ' + str(int(xyz[2][0])), (x,y), font,
                    0.5, (255, 0, 0), 2) 
        cv2.imshow('image', img) 

# reading the image 
img = cv2.imread(image_path, 1) 
  
# displaying the image 
cv2.imshow('image', img)

# setting mouse handler for the image
# and calling the click_event() function 
cv2.setMouseCallback('image', click_event)

# wait for a key to be pressed to exit 
cv2.waitKey(0) 

# close the window
cv2.destroyAllWindows()