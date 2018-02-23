import cv2
import numpy as np
import time
from utils import enlarge_roi


MEAN = (78.4263377603, 87.7689143744, 114.895847746)
SIZE = (227, 227)


class CaffeClassifier:

    def __init__(self, prototxt, caffemodel, classes):
        print("[INFO] loading model...")
        self.net = cv2.dnn.readNetFromCaffe(prototxt, caffemodel)
        self.classes = classes

    def dims(self):
        return SIZE

    def classify(self, image):
        blob = cv2.dnn.blobFromImage(image, 1.0, self.dims(), MEAN)
        self.net.setInput(blob)
        start = time.time()
        preds = self.net.forward()
        end = time.time()
        print("[INFO] classification took {:.5} seconds".format(end - start))
        # sort the probabilities (in descending) order, grab the index of the
        # top predicted label, and draw it on the input image
        idx = np.argsort(preds[0])[::-1][0]
        return self.classes[idx], preds[0][idx] * 100

    def classify_all(self, image, bboxes):
        images = []
        for bbox in bboxes:
            # print(face)
            x, y, w, h = enlarge_roi(image, bbox)
            crop = image[y:y + h, x:x + w]
            images.append(crop)

        blob = cv2.dnn.blobFromImages(images, 1.0, SIZE, MEAN, False, False)
        self.net.setInput(blob)
        start = time.time()
        preds = self.net.forward()
        end = time.time()
        print("[INFO] classification took {:.5} seconds".format(end - start))
        result = []
        for pred in preds:
            # sort the probabilities (in descending) order, grab the index of the
            # top predicted label, and draw it on the input image
            idx = np.argsort(preds[0])[::-1][0]
            result.append((self.classes[idx], pred[idx] * 100))

        return result
