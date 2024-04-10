#! /usr/bin/env bash
python3 cv-shower.py \
  -d CPU \
  -i 0 \
  -m ./object_detection/ssdlite_mobilenet_v2/FP32/ssdlite_mobilenet_v2.xml \
  -at ssd