# CAFFE for YOLO

## Reference

> You Only Look Once: Unified, Real-Time Object detection

> http://arxiv.org/abs/1506.02640

## Usage

### Data preparation
```Shell
  cd data/yolo
  ln -s /your/path/to/VOCdevkit/ .
  python ./get_list.py
  # change related path in script convert.sh
  ./convert.sh 
```

### Train
```Shell
  cd examples/yolo
  # change related path in script train.sh
  mkdir models
  nohup ./train.sh &
```

### Test
```Shell
  # if everything goes well, the map of gnet_yolo_iter_32000.caffemodel may reach ~56.
  cd examples/yolo
  ./test.sh model_path gpu_id
```
