import cv2
from threading import Thread
import datetime
import time
import sys

class FPSCounter:
    def __init__(self):
        self._start = None
        self._end = None
        self._noFrames = 0

    def start(self):
        self._start = datetime.datetime.now()
        return self

    def stop(self):
        self._end = datetime.datetime.now()

    def update(self):
        self._noFrames += 1
    
    def elapsed(self):
        return (self._end - self._start).total_seconds()

    def fps(self):
        return self._noFrames/self.elapsed()

class FrameGrabber:
    def __init__(self, src=0):
        self.vidStream = cv2.VideoCapture(src)
        
        (self.grabbed, self.frame) = self.vidStream.read()
        self.stopped = False

    def start(self):
        Thread(target=self.grabFrame, args=()).start()
        return self
        
    def grabFrame(self):
            
        while True:
            if self.stopped:
                return

            (self.grabbed, self.frame) = self.vidStream.read()
            
    def read(self):
        return self.frame

    def stop(self):
        self.vidStream.release()
        self.stopped = True
        

vidStream = FrameGrabber(src=0).start()
cascadeFace = cv2.CascadeClassifier('lbpcascade_frontalface.xml')

if (len(sys.argv) == 1):
    template = cv2.imread('template.png', 0)
else:
    imgPath = sys.argv[1]
    template = cv2.imread(imgPath, 0)

if template is None:
    #If no template file exists, open video stream to capture template
    while (True):
        tempFrame = vidStream.read()
        cv2.imshow('Template Capture', tempFrame)
        template = cv2.cvtColor(tempFrame, cv2.COLOR_BGR2GRAY)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.imwrite("template.png", template)
            break
    


w,h = template.shape[::-1]

#Reducing the template image to crop out the face
face = cascadeFace.detectMultiScale(template, scaleFactor=1.3, minNeighbors=5, minSize=(25,25))
padding = 30
for (x,y,w,h) in face:
    cv2.rectangle(template, (x,y-30), (x + w, y + h+20), (0,255,0), 2)
    cropped = template[y-30:y+h+20, x:x+w]
    

cv2.imshow('Template', template)
cv2.imshow('Cropped', cropped)
cv2.waitKey(1)

fps = FPSCounter().start()

while True:
    frame = vidStream.read()
    cv2.imshow('Frame', frame)
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    cv2.imshow('Gray', gray)
    faceCam = cascadeFace.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(25,25))
    
    for (x,y,w,h) in faceCam:
        croppedResized = cv2.resize(cropped, (w,h), interpolation=cv2.INTER_LINEAR)
        cv2.imshow('asd', croppedResized)
        mat = cv2.matchTemplate(gray, croppedResized, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(mat)
        top_left = max_loc
        bottom_right = (top_left[0] + w, top_left[1] + h + 30)
        cv2.rectangle(frame, top_left, bottom_right, (0,255,0), 2)

    time.sleep(0.001)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    fps.update()

fps.stop()
print('FPS: ', fps.fps())
print('Elapsed seconds: ', fps.elapsed())
vidStream.stop()
cv2.destroyAllWindows()
