from mtcnn.exceptions import InvalidImage
from mtcnn.mtcnn import MTCNN

from face_detectors.base import FaceDetector


class MTCNNFaceDetector(FaceDetector):

    @property
    def name(self):
        return 'MTCNN'

    def __init__(self, min_confidence):
        super(MTCNNFaceDetector, self).__init__()
        self.detector = MTCNN()
        self.min_confidence = min_confidence

    def detect(self, image, include_score=False):
        try:
            faces = [f['box'] for f in self.detector.detect_faces(image) if f['confidence'] >= self.min_confidence]
        except InvalidImage:
            faces = []
        return faces
