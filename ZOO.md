# Face detection model zoo

![Discrete ROC](images/discROC-compare.png) 

A collection of face detection models pre-trained on the [Widerface](http://mmlab.ie.cuhk.edu.hk/projects/WIDERFace/) 
dataset. [Morghulis](https://github.com/the-house-of-black-and-white/morghulis) was used to 
download and convert it to either [Darknet](https://pjreddie.com/darknet/yolo/) and [Tensorflow Object Detection API](https://github.com/tensorflow/models/tree/master/research/object_detection) format.

In the table below you can see each model detailed information including:

* a model name
* a download link to a `tar.gz` file containing the model and configuration files
* TODO model speed
* detector performance measured on the [FDDB](http://vis-www.cs.umass.edu/fddb/) benchmark

Model | mAP@0.5 | cfg/weights
--- | ---: | :---:
YOLOv2 | 84.90 |[link](https://drive.google.com/open?id=1_Uj59hkJEpht2ykZphW4m-l42odwkPJB)
Tiny YOLO | 80.04 |[link](https://drive.google.com/open?id=1koNNZv53JXzcgP_5sPMUVlAnB7HW8uLc)
SSD mobilenet v1 | ? |[link](https://drive.google.com/open?id=1NT3PLBHa4cYj_RmKlRrCZSWMKMct2-26)
Faster RCNN inception resnet v2 atrous | 94.39 |[link](https://drive.google.com/open?id=1bMdKHMcVidrG7BUvoIk6cCcEGKhBFvcc)
R-FCN resnet101 | 94.73 |[link](https://drive.google.com/open?id=1is7Ldv9ASYNcrv2GyXS7EaV58UaqhuFQ)


## Training details

### Darknet

There are 2 models trained with [Darknet](https://pjreddie.com/darknet/yolo/): one based on YOLOv2 and other 
on Tiny YOLO. Both used convolutional weights that are pre-trained on Imagenet: 
[darknet19_448.conv.23](https://pjreddie.com/media/files/darknet19_448.conv.23).

### Tensorflow Object Detection API

The remaining models were trained with [Tensorflow Object Detection API](https://github.com/tensorflow/models/tree/master/research/object_detection) 
on [Google Cloud ML Engine](https://cloud.google.com/ml-engine/docs/technical-overview).


## Face detectors performance **evaluation** on the FDDB dataset

### Discrete ROC

![Discrete ROC](images/discROC-compare.png) 


### Continuous ROC

![Continuous ROC](images/contROC-compare.png) 



## Face detection using YOLOv2

This demo shows 2 face detectors:

1) OpenCVs Viola Jones implementation (red)
2) YOLOv2 trained on the WIDER FACE dataset (green)

It's using  [OpenCV's dnn module for YOLO inference](https://github.com/opencv/opencv/pull/9705).
Check out the video below:

[![Demo](http://img.youtube.com/vi/dkTi8naw67Y/0.jpg)](http://www.youtube.com/watch?v=dkTi8naw67Y)

### Running

You can easily run this demo by cloning this repo and running the `run.sh` script, or with docker directly:

```bash
xhost + && \
docker run --privileged --rm -it \
  -e DISPLAY=unix$DISPLAY \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
   housebw/demo python app.py
```

The `housebw/demo` docker image includes all necessary dependencies as well as the trained models.
