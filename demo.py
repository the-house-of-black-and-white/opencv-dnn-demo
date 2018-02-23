import argparse

import cv2
from imutils.video import FPS
from imutils.video import WebcamVideoStream

from demographics.age import AgeClassifier
from demographics.gender import GenderClassifier
from face_detectors import new_face_detector
from utils import enlarge_roi

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-src', '--source', dest='video_source', type=int,
                        help='Video file name.')
    parser.add_argument('-fd', '--face-detector', dest='detector', type=str,
                        default='yolo', help='yolo, mtcnn or vj')
    parser.add_argument('-thresh', '--threshold', dest='min_confidence', type=float,
                        default=0.5, help='Min confidence threshold.')
    parser.add_argument('-codec', '--codec', dest='codec', type=str,
                        default='XVID', help='codec MJPG or XVID')
    parser.add_argument('-save', '--save', dest='save', type=str,
                        default='output', help='Save video.')
    parser.add_argument('-fps', '--fps', dest='fps', type=float,
                        default=30, help='FPS.')

    args = parser.parse_args()

    print("[INFO] starting webcam...")
    fvs = WebcamVideoStream(args.video_source).start()
    face_detector = new_face_detector(args.detector, args.min_confidence)
    age_classifier = AgeClassifier()
    gender_classifier = GenderClassifier()
    # gender_classifier = DexGenderClassifier()
    # time.sleep(1.0)
    # Define the codec and create VideoWriter object
    # fourcc = cv2.VideoWriter_fourcc(*args.codec)
    # writer = None
    # start the FPS timer
    fps = FPS().start()

    while True:
        frame = fvs.read()
        # check if the writer is None
        # if writer is None:
        #     # store the image dimensions, initialize the video writer
        #     (h, w) = frame.shape[:2]
        #     writer = cv2.VideoWriter('{}_{}_{}.avi'.format(args.save, args.detector, int(time.time())), fourcc, args.fps, (w, h), True)
        faces, ages, genders = face_detector.detect(frame), [], []

        if len(faces) > 0:
            ages = age_classifier.classify_all(frame, faces)
            genders = gender_classifier.classify_all(frame, faces)

        for face, age, gender in zip(faces, ages, genders):
            print(face, age, gender)
            x, y, w, h = enlarge_roi(frame, face)

            # bbox face
            # label = face_detector.name
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            age_score = age[1]
            age_label = age[0]
            if age_score > args.min_confidence:
                label = "{}: {:.2f}%".format(age_label, age_score)
                label_size, base_line = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(frame, (x, y), (x + label_size[0], y + label_size[1] + base_line), (0, 255, 0), cv2.FILLED)
                cv2.putText(frame, label, (x, y + label_size[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

            gender_label = gender[0]
            gender_score = gender[1]
            if gender_score > args.min_confidence:
                label = "{}: {:.2f}%".format(gender_label, gender_score)
                label_size, base_line = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(frame, (x + w - label_size[0] - base_line, y + h - label_size[1]), (x + w, y + h), (0, 255, 0), cv2.FILLED)
                cv2.putText(frame, label, (x + w - label_size[0], y + h), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

        # writer.write(frame)
        cv2.imshow(face_detector.name, frame)
        fps.update()
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # stop the timer and display FPS information
    fps.stop()
    print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
    print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

    # do a bit of cleanup
    cv2.destroyAllWindows()
    # writer.release()
    fvs.stop()
