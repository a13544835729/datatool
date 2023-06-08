import os
import pandas as pd
import numpy as np
from shutil import copy2
import json
import torch
from tqdm import tqdm
import cv2


def xyxy2xywh(x):
    # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = (x[:, 0] + x[:, 2]) / 2  # x center
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2  # y center
    y[:, 2] = x[:, 2] - x[:, 0]  # width
    y[:, 3] = x[:, 3] - x[:, 1]  # height
    return y




def bound2box(boundary:list,file='',w=2560,h=2560):
    txtlist=[]
    for i,bouny in enumerate(boundary):
        bound=bouny['boundary']
        # print(bound)
        df=pd.DataFrame(bound)
        x_min=df['x'].min()
        y_min=df['y'].min()
        x_max=df['x'].max()
        y_max=df['y'].max()
        label = 0
        temp = [int(label), x_min/w, y_min/h, x_max/w, y_max/h]
        # temp = [label, x_min, y_min, x_max, y_max]
        txtlist.append(temp)
    xyxy=np.array(txtlist)
    # print(xyxy)
    xywh=xyxy.copy()
    # xywh[:,1:]=xyxy2xywh(xyxy[:,1:])
    xywh=pd.DataFrame(xywh)

    xywh.to_csv(file,index=None,header=None,sep=' ')

    # xyxy = pd.DataFrame(xyxy)
    # xyxy.to_csv(file,index=None,header=None)




def bound2Rect(boundary:list,savefile='',cls='grains',cls_num=0):
    boxxes=[]
    for bound in boundary:
        cnt=pd.DataFrame(bound['boundary'])[['x','y']].values
        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)
        box = box.astype('int')
        box = box.flatten().tolist()
        box.extend([cls,cls_num])
        boxxes.append(box)

    boxdf=pd.DataFrame(boxxes)
    boxdf.to_csv(savefile,header=None,sep=' ',index=None)






def poly2minAreaRect(ploysrc='',retdst='',cls=''):
    namelist=os.listdir(ploysrc)
    for i,name in tqdm(enumerate(namelist)):
        print(name)
        srcpth = os.path.join(ploysrc, name)
        dstpth = os.path.join(retdst, name[:-4]+'.txt')
        # imgpth = os.path.join(imgsrc,name[:-4]+'.png')

        with open(srcpth,'r') as f:
            datajson = json.load(f)
            if isinstance(datajson,list):
                for data in datajson:

                    boundarys=data['boundarys']
                    bound2Rect(boundarys,dstpth,cls=cls)

            else:
                boundarys = datajson['boundarys']
                bound2Rect(boundarys, dstpth, cls=cls)


def poly2minRect(ploysrc='',retdst='',cls=''):
    namelist=os.listdir(ploysrc)
    for i,name in tqdm(enumerate(namelist)):
        print(name)
        srcpth = os.path.join(ploysrc, name)
        dstpth = os.path.join(retdst, name[:-4]+'.txt')
        # imgpth = os.path.join(imgsrc,name[:-4]+'.png')

        with open(srcpth,'r') as f:
            datajson = json.load(f)
            if isinstance(datajson,list):
                for data in datajson:

                    boundarys=data['boundarys']
                    bound2box(boundarys,dstpth)
            else:
                boundarys = datajson['boundarys']
                bound2box(boundarys, dstpth)
        # break




if __name__ == '__main__':
    polysrc='/home/jasonwang/Documents/worktable/XAG/dataset/wire/电线分割标注第一批-V1已标注数据-2023年05月26日15时37分01秒'
    retdst='/home/jasonwang/Documents/worktable/XAG/dataset/rice/阶段一/yolo'
    cls='rice'
    poly2minAreaRect(ploysrc=polysrc,retdst=retdst,cls=cls)

