from pycocotools.coco import COCO
import matplotlib.pyplot as plt
import cv2
import os
import numpy as np
import random



annFile='/home/jasonwang/Documents/worktable/XAG/dataset/seed/seed_coco/annotations/train.json'

# # initialize COCO api for instance annotations
coco=COCO(annFile)
# 利用getCatIds函数获取某个类别对应的ID，
# 这个函数可以实现更复杂的功能，请参考官方文档
ids = coco.getCatIds('seeding')[0]
imgIds = coco.catToImgs[ids]
print(imgIds)

