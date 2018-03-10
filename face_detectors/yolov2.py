import cv2

from face_detectors.base import FaceDetector

inWidth = 544
inHeight = 544
inScaleFactor = 1 / float(255)


class YOLOV2FaceDetector(FaceDetector):

    @property
    def name(self):
        return 'YOLOv2'

    def __init__(self, min_confidence):
        super(YOLOV2FaceDetector, self).__init__()
        self.min_confidence = min_confidence
        self.cfg = 'models/yolo/yolo-obj.cfg'
        self.model = 'models/yolo/yolo-face_1400.weights'
        self.names = 'models/yolo/obj.names'
        self.net = cv2.dnn.readNetFromDarknet(self.cfg, self.model)
        if self.net.empty():
            exit(1)

    def detect(self, image, include_score=False):
        blob = cv2.dnn.blobFromImage(image, inScaleFactor, (inWidth, inHeight), (0, 0, 0), True, False)
        self.net.setInput(blob)
        detections = self.net.forward()
        rows, cols, _ = image.shape
        faces = []
        for i in range(detections.shape[0]):
            confidence = detections[i, 5]
            if confidence > self.min_confidence:
                x_center = detections[i, 0] * cols
                y_center = detections[i, 1] * rows
                width = detections[i, 2] * cols
                height = detections[i, 3] * rows
                xmin = int(round(x_center - width / 2))
                ymin = int(round(y_center - height / 2))
                # xmax = int(round(x_center + width / 2))
                # ymax = int(round(y_center + height / 2))
                if include_score:
                    faces.append([xmin, ymin, int(width), int(height), confidence])
                else:
                    faces.append([xmin, ymin, int(width), int(height)])
        return faces
