#!/usr/bin/env python
import numpy as np
import cv2
import os, sys
sys.path.insert(0, '../../python/')
import caffe
caffe.set_device(0)
caffe.set_mode_gpu()

mean = np.require([104, 117, 123], dtype=np.float32)[:, np.newaxis, np.newaxis]

def nms(dets, thresh):
  # -------------------------
  # Pure Python NMS baseline.
  # Written by Ross Girshick
  # -------------------------
  x1 = dets[:, 0] - dets[:, 2] / 2.
  y1 = dets[:, 1] - dets[:, 3] / 2.
  x2 = dets[:, 0] + dets[:, 2] / 2.
  y2 = dets[:, 1] + dets[:, 3] / 2.
  scores = dets[:, 4]
  areas = (x2 - x1 + 1) * (y2 - y1 + 1)
  order = scores.argsort()[::-1]
  keep = []
  while order.size > 0:
    i = order[0]
    keep.append(i)
    xx1 = np.maximum(x1[i], x1[order[1:]])
    yy1 = np.maximum(y1[i], y1[order[1:]])
    xx2 = np.minimum(x2[i], x2[order[1:]])
    yy2 = np.minimum(y2[i], y2[order[1:]])
    w = np.maximum(0.0, xx2 - xx1 + 1)
    h = np.maximum(0.0, yy2 - yy1 + 1)
    inter = w * h
    ovr = inter / (areas[i] + areas[order[1:]] - inter)
    inds = np.where(ovr <= thresh)[0]
    order = order[inds + 1]
  return dets[np.require(keep), :]

def parse_result(reg_out):
  num_classes = 20
  num_objects = 2
  side = 7
  locations = side ** 2
  boxes = np.zeros((num_objects * locations, 6), dtype=np.float32)
  for i in range(locations):
    tmp_scores = reg_out[i:num_classes*locations:locations]
    max_class_ind = np.argsort(tmp_scores)[-1]
    max_prob = np.max(tmp_scores)
    obj_index = num_classes * locations + i
    obj_scores = max_prob * reg_out[obj_index:(obj_index+num_objects*locations):locations]
    coor_index = (num_classes + num_objects) * locations + i
    for j in range(num_objects):
      boxes[i*num_objects+j][5] = max_class_ind
      boxes[i*num_objects+j][4] = obj_scores[j]
      box_index = coor_index + j * 4 * locations
      boxes[i*num_objects+j][0] = (i % side + reg_out[box_index + 0 * locations]) / float(side)
      boxes[i*num_objects+j][1] = (i / side + reg_out[box_index + 1 * locations]) / float(side)
      boxes[i*num_objects+j][2] = reg_out[box_index + 2 * locations] ** 2
      boxes[i*num_objects+j][3] = reg_out[box_index + 3 * locations] ** 2
  return nms(boxes, 0.7)

def show_boxes(im_path, boxes, thresh=0.5, show=0):
  print boxes.shape
  im = cv2.imread(im_path)
  ori_w = im.shape[1]
  ori_h = im.shape[0]
  for box in boxes:
    if box[4] < thresh:
      continue
    print box
    box = box[:4]
    x1 = max(0, int((box[0] - box[2] / 2.) * ori_w))
    y1 = max(0, int((box[1] - box[3] / 2.) * ori_h))
    x2 = min(ori_w - 1, int((box[0] + box[2] / 2.) * ori_w))
    y2 = min(ori_h - 1, int((box[1] + box[3] / 2.) * ori_h))
    cv2.rectangle(im, (x1, y1), (x2, y2), (0, 0, 255), 2)
  if show:
    cv2.imshow("out", im)
  else:
    cv2.imwrite("out.jpg", im)

def det(model, im_path, show=0):
  '''forward processing'''
  im = cv2.imread(im_path)
  im = cv2.resize(im, (448, 448))
  im = np.require(im.transpose((2, 0, 1)), dtype=np.float32)
  im -= mean
  model.blobs['data'].data[...] = im
  out_blobs = model.forward()
  reg_out = out_blobs["regression"]
  boxes = parse_result(reg_out[0])
  show_boxes(im_path, boxes, 0.2)

if __name__=="__main__":
  net_proto = "./gnet_deploy.prototxt"
  model_path = "./models/gnet_yolo_iter_32000.caffemodel"
  model = caffe.Net(net_proto, model_path, caffe.TEST)

  im_path = sys.argv[1]
  det(model, im_path)

