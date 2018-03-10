# Face detection using YOLOv2

This demo shows 2 face detectors:

1) OpenCVs Viola Jones implementation (red)
2) YOLOv2 trained on the WIDER FACE dataset (green)

It's using  [OpenCV's dnn module for YOLO inference](https://github.com/opencv/opencv/pull/9705).
Check out the video below:

[![Demo](http://img.youtube.com/vi/dkTi8naw67Y/0.jpg)](http://www.youtube.com/watch?v=dkTi8naw67Y)

## Running

You can easily run this demo by cloning this repo and running the `run.sh` script, or with docker directly:

```bash
xhost + && \
docker run --privileged --rm -it \
  -e DISPLAY=unix$DISPLAY \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
   housebw/demo python app.py
```

The `housebw/demo` docker image includes all necessary dependencies as well as the trained models.

## Face Detection Model Zoo

Check out the 5 pre trained models in the [zoo](ZOO.md)!


