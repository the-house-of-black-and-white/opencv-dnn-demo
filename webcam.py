import argparse
import multiprocessing
from multiprocessing import Queue, Pool

import cv2 as cv

from face_detectors import new_face_detector
from utils import FPS, WebcamVideoStream, enlarge_roi


def detect_objects(img, face_detector):
    faces = face_detector.detect(img)
    for face in faces:
        x, y, w, h = enlarge_roi(img, face)
        # Blur Face
        sub_face = img[y:y + h, x:x + w]
        sub_face = cv.GaussianBlur(sub_face, (103, 103), 100)
        print(sub_face.shape)
        # merge this blurry rectangle to our final image
        img[y:y + h, x:x + w] = sub_face

        # bbox face
        # label = face_detector.name
        # cv.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        # label_size, base_line = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        # cv.rectangle(img, (x, y), (x + label_size[0], y + label_size[1] + base_line), (0, 255, 0), cv.FILLED)
        # cv.putText(img, label, (x, y + label_size[1]), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

        # cv.rectangle(img, (x - PADDING, y - PADDING), (x + w + PADDING, y + h + PADDING), (0, 0, 0), cv.FILLED)

    return img


def worker(input_q, output_q, detector):
    face_detector = new_face_detector(detector)
    while True:
        frm = input_q.get()
        frm = detect_objects(frm, face_detector)
        output_q.put(frm)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-src', '--source', dest='video_source', type=int,
                        default=1, help='Device index of the camera.')
    parser.add_argument('-wd', '--width', dest='width', type=int,
                        default=640, help='Width of the frames in the video stream.')
    parser.add_argument('-ht', '--height', dest='height', type=int,
                        default=480, help='Height of the frames in the video stream.')
    parser.add_argument('-num-w', '--num-workers', dest='num_workers', type=int,
                        default=1, help='Number of workers.')
    parser.add_argument('-q-size', '--queue-size', dest='queue_size', type=int,
                        default=10, help='Size of the queue.')
    parser.add_argument('-thresh', '--threshold', dest='min_confidence', type=float,
                        default=0.5, help='Min confidence threshold.')
    parser.add_argument('-fps', '--fps', dest='fps', type=float,
                        default=3.5, help='FPS.')
    parser.add_argument('-codec', '--codec', dest='codec', type=str,
                        default='XVID', help='codec MJPG or XVID')
    parser.add_argument('-save', '--save', dest='save', type=str,
                        default='output', help='Save video.')
    parser.add_argument('-fd', '--face-detector', dest='detector', type=str,
                        default='yolo', help='yolo, mtcnn or vj')

    args = parser.parse_args()

    logger = multiprocessing.log_to_stderr()
    logger.setLevel(multiprocessing.SUBDEBUG)
    input_q = Queue(maxsize=args.queue_size)
    output_q = Queue(maxsize=args.queue_size)

    # args.min_confidence
    detector = args.detector
    pool = Pool(args.num_workers, worker, (input_q, output_q, detector))
    video_capture = WebcamVideoStream(src=args.video_source,
                                      width=args.width,
                                      height=args.height).start()

    # Define the codec and create VideoWriter object
    fourcc = cv.VideoWriter_fourcc(*args.codec)
    out = cv.VideoWriter('{}_{}.avi'.format(args.save, detector), fourcc, args.fps, (args.width, args.height))

    fps = FPS().start()
    while True:
        frame = video_capture.read()
        # Start timer
        timer = cv.getTickCount()
        input_q.put(frame)
        output_frame = output_q.get()
        fps.update()
        # Calculate Frames per second (FPS)
        _fps = cv.getTickFrequency() / (cv.getTickCount() - timer)
        # Display FPS on frame
        # cv.putText(output_frame, "FPS : " + str(int(_fps)), (100, 50), cv.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2)
        out.write(output_frame)
        cv.imshow('Video', output_frame)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    fps.stop()
    print('[INFO] elapsed time (total): {:.2f}'.format(fps.elapsed()))
    print('[INFO] approx. FPS: {:.2f}'.format(fps.fps()))

    pool.terminate()
    video_capture.stop()
    out.release()
    cv.destroyAllWindows()
