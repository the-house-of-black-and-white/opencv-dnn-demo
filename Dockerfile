FROM python:2

RUN pip install opencv-python

WORKDIR /usr/src/app

ADD . .
