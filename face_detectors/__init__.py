from face_detectors.mtcnn import MTCNNFaceDetector
from face_detectors.violajones import ViolaJonesFaceDetector
from face_detectors.yolov2 import YOLOV2FaceDetector


def new_face_detector(fd_type, min_confidence=0.5):

    if fd_type == 'yolo':
        detector = YOLOV2FaceDetector(min_confidence)
    elif fd_type == 'mtcnn':
        detector = MTCNNFaceDetector(min_confidence)
    elif fd_type == 'vj':
        detector = ViolaJonesFaceDetector()
    else:
        raise ValueError()

    return detector


