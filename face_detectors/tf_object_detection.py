from abc import ABCMeta

import numpy as np
import tensorflow as tf

from face_detectors.base import FaceDetector


class BaseTensorflowFaceDetector(FaceDetector):
    __metaclass__ = ABCMeta

    def __init__(self, min_confidence, checkpoint):
        super(BaseTensorflowFaceDetector, self).__init__()
        self.min_confidence = min_confidence
        self.detection_graph = tf.Graph()
        self.checkpoint = checkpoint
        with self.detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(self.checkpoint, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')
            self.sess = tf.Session(graph=self.detection_graph)

    def detect(self, image, include_score=False):
        # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
        image_np_expanded = np.expand_dims(image, axis=0)
        image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')

        # Each box represents a part of the image where a particular object was detected.
        boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')

        # Each score represent how level of confidence for each of the objects.
        # Score is shown on the result image, together with the class label.
        scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
        classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
        num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')

        # Actual detection.
        (boxes, scores, classes, num_detections) = self.sess.run(
            [boxes, scores, classes, num_detections],
            feed_dict={image_tensor: image_np_expanded})

        boxes = np.squeeze(boxes)
        scores = np.squeeze(scores)
        faces = []
        im_height, im_width, _ = image.shape
        for i in range(boxes.shape[0]):
            if scores[i] >= self.min_confidence:
                ymin, xmin, ymax, xmax = tuple(boxes[i].tolist())
                (left, right, top, bottom) = (xmin * im_width, xmax * im_width,
                                              ymin * im_height, ymax * im_height)

                if include_score:
                    faces.append([int(left), int(top), int(right - left), int(bottom - top), scores[i]])
                else:
                    faces.append([int(left), int(top), int(right - left), int(bottom - top)])

        return faces


class RfcnResnet101FaceDetector(BaseTensorflowFaceDetector):

    @property
    def name(self):
        return 'RFCN'

    def __init__(self, min_confidence, checkpoint='models/tf/rfcn_resnet101_91674/frozen_inference_graph.pb'):
        super(RfcnResnet101FaceDetector, self).__init__(min_confidence, checkpoint=checkpoint)


class FasterRCNNFaceDetector(BaseTensorflowFaceDetector):

    @property
    def name(self):
        return 'fasterRCNN'

    def __init__(self, min_confidence, checkpoint='models/tf/faster_rcnn_inception_resnet_v2_atrous_65705/frozen_inference_graph.pb'):
        super(FasterRCNNFaceDetector, self).__init__(min_confidence, checkpoint=checkpoint)


class SSDMobileNetV1FaceDetector(BaseTensorflowFaceDetector):

    @property
    def name(self):
        return 'ssd'

    def __init__(self, min_confidence, checkpoint='models/tf/ssd_mobilenet_v1_106650/frozen_inference_graph.pb'):
        super(SSDMobileNetV1FaceDetector, self).__init__(min_confidence, checkpoint=checkpoint)
