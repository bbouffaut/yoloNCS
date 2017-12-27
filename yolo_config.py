from easydict import EasyDict as edict
import os

__C = edict()
cfg = __C

real_path = os.path.dirname(os.path.realpath(__file__))
__C.NCS_GRAPH = os.path.join(real_path, "graph")
__C.MODEL_INPUT_SIZE = (448,448)
__C.CLASSES = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train","tvmonitor"]
__C.BOXES_FILTERING_THRESHOLD = 0.2
__C.IOU_FILTERING_THRESHOLD = 0.5
__C.NUM_CLASSES = len(__C.CLASSES)
__C.NUM_BOXES = 2
__C.GRID_SIZE = 7
