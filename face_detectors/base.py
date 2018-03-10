from abc import ABCMeta, abstractmethod, abstractproperty


class FaceDetector:
    __metaclass__ = ABCMeta

    def __init__(self):
        pass

    @abstractmethod
    def detect(self, image, include_score=False):
        pass

    @abstractproperty
    def name(self):
        pass