import argparse

import cv2
import time
from imutils.video import FPS
from imutils.video import FileVideoStream

from face_detectors import new_face_detector
from utils import enlarge_roi

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-src', '--source', dest='video_source', type=str,
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

    # start the file video stream thread and allow the buffer to
    # start to fill
    print("[INFO] starting video file thread...")
    fvs = FileVideoStream(args.video_source).start()
    face_detector = new_face_detector(args.detector, args.min_confidence)
    time.sleep(1.0)
    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*args.codec)
    writer = None

    # start the FPS timer
    fps = FPS().start()

    # loop over frames from the video file stream
    while fvs.more():
        frame = fvs.read()

        # check if the writer is None
        if writer is None:
            # store the image dimensions, initialize the video writer
            (h, w) = frame.shape[:2]
            writer = cv2.VideoWriter('{}_{}_{}.avi'.format(args.save, args.detector, int(time.time())), fourcc, args.fps, (w, h), True)

        faces = face_detector.detect(frame)
        for face in faces:
            x, y, w, h = enlarge_roi(frame, face)
            sub_face = frame[y:y + h, x:x + w]
            sub_face = cv2.GaussianBlur(sub_face, (103, 103), 100)
            frame[y:y + h, x:x + w] = sub_face

        writer.write(frame)
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
    writer.release()
    fvs.stop()



