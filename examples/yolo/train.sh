#!/usr/bin/env sh

CAFFE_HOME=../..

SOLVER=./gnet_solver.prototxt
WEIGHTS=/your/path/to/bvlc_googlenet.caffemodel

$CAFFE_HOME/build/tools/caffe train \
    --solver=$SOLVER --weights=$WEIGHTS --gpu=0,1

