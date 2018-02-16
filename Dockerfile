FROM tensorflow/tensorflow:1.5.0-gpu-py3

RUN apt-get update && \
    apt-get install -y \
    libsm6 \
    libxext6 \
    libxrender1 \
    libfontconfig1

RUN pip install opencv-contrib-python mtcnn imutils

WORKDIR /usr/src/app

ADD . .
