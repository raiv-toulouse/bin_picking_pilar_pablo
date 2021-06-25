import cv2
import threading

class webcamImageGetter:

    def __init__(self):
        self.currentFrame = None
        self.CAMERA_WIDTH = 1280
        self.CAMERA_HEIGHT = 960
        self.CAMERA_NUM = 0

        self.capture = cv2.VideoCapture(0) #Put in correct capture number here
        #OpenCV by default gets a half resolution image so we manually set the correct resolution
        self.capture.set(3, self.CAMERA_WIDTH)
        self.capture.set(4, self.CAMERA_HEIGHT)

    #Starts updating the images in a thread
    def start(self):
        threading.Thread(target=self.updateFrame, args=()).start()

    #Continually updates the frame
    def updateFrame(self):
        while(True):
            ret, self.currentFrame = self.capture.read()
            cv2.imshow("2 frame", self.currentFrame)

            while (self.currentFrame == None): #Continually grab frames until we get a good one
                ret, frame = self.capture.read()

    def getFrame(self):
        return self.currentFrame