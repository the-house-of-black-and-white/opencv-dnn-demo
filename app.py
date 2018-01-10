import argparse
import multiprocessing
import time
from multiprocessing import Queue, Pool

import cv2 as cv

from utils import FPS, WebcamVideoStream

inWidth = 416
inHeight = 416
inScaleFactor = 1 / float(255)

face_cascade = cv.CascadeClassifier(cv.data.haarcascades + "haarcascade_frontalface_default.xml")


def detect_face_vj(frame):
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    cv.equalizeHist(gray, gray)
    faces = face_cascade.detectMultiScale(gray, 1.3, 3)
    for (x, y, w, h) in faces:
        cv.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
        label = 'ViolaJones'
        label_size, base_line = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv.rectangle(frame, (x, y), (x + label_size[0], y + label_size[1] + base_line), (0, 0, 255),
                     cv.FILLED)
        cv.putText(frame, label, (x, y + label_size[1]), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))
    return frame


def detect_objects(frame, net, min_confidence):
    blob = cv.dnn.blobFromImage(frame, inScaleFactor, (inWidth, inHeight), (0, 0, 0), True, False)
    net.setInput(blob)
    detections = net.forward()
    rows = frame.shape[0]
    cols = frame.shape[1]
    for i in range(detections.shape[0]):
        confidence = detections[i, 5]
        if confidence > min_confidence:
            x_center = detections[i, 0] * cols
            y_center = detections[i, 1] * rows
            width = detections[i, 2] * cols
            height = detections[i, 3] * rows

            xmin = int(round(x_center - width / 2))
            ymin = int(round(y_center - height / 2))
            xmax = int(round(x_center + width / 2))
            ymax = int(round(y_center + height / 2))

            cv.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)

            label = "DNN: " + str(confidence)
            label_size, base_line = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv.rectangle(frame, (xmin, ymin), (xmin + label_size[0], ymin + label_size[1] + base_line), (0, 255, 0),
                         cv.FILLED)
            cv.putText(frame, label, (xmin, ymin + label_size[1]), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

    return frame


def worker(input_q, output_q, net, min_confidence):
    fps = FPS().start()
    while True:
        fps.update()
        frm = input_q.get()
        frm = detect_objects(frm, net, min_confidence)
        frm = detect_face_vj(frm)
        output_q.put(frm)
    fps.stop()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-src', '--source', dest='video_source', type=int,
                        default=1, help='Device index of the camera.')
    parser.add_argument('-wd', '--width', dest='width', type=int,
                        default=640, help='Width of the frames in the video stream.')
    parser.add_argument('-ht', '--height', dest='height', type=int,
                        default=480, help='Height of the frames in the video stream.')
    parser.add_argument('-num-w', '--num-workers', dest='num_workers', type=int,
                        default=4, help='Number of workers.')
    parser.add_argument('-q-size', '--queue-size', dest='queue_size', type=int,
                        default=10, help='Size of the queue.')
    parser.add_argument('-thresh', '--threshold', dest='min_confidence', type=float,
                        default=0.5, help='Min confidence threshold.')
    parser.add_argument('-fps', '--fps', dest='fps', type=float,
                        default=3.5, help='FPS.')
    parser.add_argument('-codec', '--codec', dest='codec', type=str,
                        default='XVID', help='codec MJPG or XVID')
    parser.add_argument('-save', '--save', dest='save', type=str,
                        default='output.avi', help='Save video.')
    args = parser.parse_args()

    logger = multiprocessing.log_to_stderr()
    logger.setLevel(multiprocessing.SUBDEBUG)
    input_q = Queue(maxsize=args.queue_size)
    output_q = Queue(maxsize=args.queue_size)

    # cfg = 'models/tiny-yolo/tiny-yolo-obj.cfg'
    # model = 'models/tiny-yolo/tiny-yolo-face_13000.weights'

    cfg = 'models/yolo/yolo-obj.cfg'
    model = 'models/yolo/yolo-face_1400.weights'
    names = 'models/yolo/obj.names'
    net = cv.dnn.readNetFromDarknet(cfg, model)
    if net.empty():
        exit(1)

    pool = Pool(args.num_workers, worker, (input_q, output_q, net, args.min_confidence))

    video_capture = WebcamVideoStream(src=args.video_source,
                                      width=args.width,
                                      height=args.height).start()

    # Define the codec and create VideoWriter object
    fourcc = cv.VideoWriter_fourcc(*args.codec)
    out = cv.VideoWriter(args.save, fourcc, args.fps, (args.width, args.height))

    fps = FPS().start()
    while True:  # fps._numFrames < 120
        frame = video_capture.read()
        input_q.put(frame)
        t = time.time()
        output_frame = output_q.get()
        out.write(output_frame)
        cv.imshow('Video', output_frame)
        fps.update()
        print('[INFO] elapsed time: {:.2f}'.format(time.time() - t))
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    fps.stop()
    print('[INFO] elapsed time (total): {:.2f}'.format(fps.elapsed()))
    print('[INFO] approx. FPS: {:.2f}'.format(fps.fps()))

    pool.terminate()
    video_capture.stop()
    out.release()
    cv.destroyAllWindows()
