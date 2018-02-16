from abc import ABCMeta, abstractmethod, abstractproperty


class FaceDetector:
    __metaclass__ = ABCMeta

    def __init__(self):
        pass

    @abstractmethod
    def detect(self, image):
        pass

    @abstractproperty
    def name(self):
        pass