#!/bin/bash

xhost +

docker run --privileged --rm -it \
  -e DISPLAY=unix$DISPLAY \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -v $PWD:/usr/src/app \
   housebw/demo python app.py

xhost -