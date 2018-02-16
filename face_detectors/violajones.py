import cv2 as cv
from face_detectors.base import FaceDetector


class ViolaJonesFaceDetector(FaceDetector):

    @property
    def name(self):
        return 'ViolaJones'

    def __init__(self):
        super(ViolaJonesFaceDetector, self).__init__()
        self.scale_factor = 1.1
        self.min_neighbors = 3
        self.min_size = (30, 30)
        self.face_cascade = cv.CascadeClassifier(cv.data.haarcascades + "haarcascade_frontalface_default.xml")

    def detect(self, frame):
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        cv.equalizeHist(gray, gray)
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=self.scale_factor, minNeighbors=self.min_neighbors,
                                              minSize=self.min_size, flags=cv.CASCADE_SCALE_IMAGE)
        return faces
