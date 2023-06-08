import os
import cv2
import requests
import json
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm



def folder_check(folder):
    """检测文件夹是否存在"""
    if not os.path.exists(folder):
        os.makedirs(folder)


def xyxy2xywh(x):
    # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = (x[:, 0] + x[:, 2]) / 2  # x center
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2  # y center
    y[:, 2] = x[:, 2] - x[:, 0]  # width
    y[:, 3] = x[:, 3] - x[:, 1]  # height
    return y


def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y


def url2img(url,file='./image/img2.png'):
    html=requests.get(url=url)
    with open(file, 'wb') as f:
        f.write(html.content)




def json2txt(datalist:list,file='test.txt'):

    txtlist=[]
    for data in datalist:
        x_max=data['x_max']
        y_max=data['y_max']
        x_min=data['x_min']
        y_min=data['y_min']
        label=0
        temp=[label,x_min,y_min,x_max,y_max]
        txtlist.append(temp)
        for item in temp:
            f=open(file,'a')
            f.write(str(item))
            f.write(' ')
        f.write('\n')
    f.close()



def json2polytxt(data,jsonpth):

    with open(jsonpth, 'w') as f:
        json.dump(data, f,ensure_ascii=False)


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



#直接画多边形框
def json2linetxt(objects,file='test.txt'):
    print(objects)
    boxxes = []
    for obj in objects:
        boundary = obj['boundary']
        labelName = 'wire'
        label = 0
        cnt = pd.DataFrame(boundary)[['x', 'y']].values
        rect = cv2.minAreaRect(cnt)
        templist = list(rect[1])
        minindex = templist.index(min(templist))
        templist[minindex] += 15
        rect = (rect[0],tuple(templist),rect[2])
        # print(rect)
        box = cv2.boxPoints(rect)
        # box = box.astype('int')
        box = box.flatten().tolist()
        box = [round(bo,2) for bo in box]
        box.extend([labelName, label])
        boxxes.append(box)

    boxdf = pd.DataFrame(boxxes)
    boxdf.to_csv(file, header=None, sep=' ', index=None)

# 固定宽度
def json2fixedline(objects,file='test.txt'):
    print(objects)
    boxxes = []
    for obj in objects:
        boundary = obj['boundary']
        labelName = 'wire'
        label = 0
        for i in range(len(boundary)-1):
            line_start = np.array([boundary[i]['x'],boundary[i]['y']])
            line_end = np.array([boundary[i+1]['x'],boundary[i+1]['y']])
            points = line_to_rectangle(line_start,line_end,rectangle_height=15)
            points = points.flatten().tolist()
            points = [round(point,3) for point in points]
            points.append(labelName)
            points.append(label)
            boxxes.append(points)

    boxdf = pd.DataFrame(boxxes)
    boxdf.to_csv(file, header=None, sep=' ', index=None)



# 固定宽度
def line_to_rectangle(line_start, line_end,rectangle_height=1):
    line_length = np.linalg.norm(line_end - line_start)  # 计算线段的长度
    line_angle = np.arctan2(line_end[1] - line_start[1], line_end[0] - line_start[0])  # 计算线段的角度

    # 创建矩形框
    rectangle_width = line_length  # 宽度设置为线段的长度
    # rectangle_height = 1 # 根据需求设置矩形框的高度
    rectangle_center = (line_start + line_end) / 2  # 位置设置为线段的中点
    rectangle_angle = np.degrees(line_angle)  # 将角度转换为度数
    points = cv2.boxPoints((tuple(rectangle_center), (rectangle_width, rectangle_height), rectangle_angle))


    return points




def line_to_coco(objects, annfile):
    boxxes = []
    for obj in objects:
        boundary = obj['boundary']
        labelName = 'wire'
        label = 0
        cnt = pd.DataFrame(boundary)[['x', 'y']].values
        rect = cv2.minAreaRect(cnt)
        print(rect)



def loadLineImageAndLabel(datajson,annpth='',imgpth=''):
    # images
    imageName = datajson['name']
    imageUrl = datajson['image_url']
    objects = datajson['object']

    imgfile = os.path.join(imgpth, imageName+'.jpg')
    url2img(url=imageUrl, file=imgfile)

    annfile = os.path.join(annpth, imageName+'.txt')
    # json2linetxt(objects, annfile)
    # json2fixedline(objects, annfile)
    line_to_coco(objects,annfile)




def loadImageAndLabel(datajson,annpth='',imgpth=''):
    #images
    imageName = datajson['image_name']
    imageUrl = datajson['image_url']

    #labels
    markerName = datajson['markerName']
    marker = datajson['marker']
    object = marker['object']
    image_height = marker['image_height']
    image_width = marker['image_width']

    imgfile = os.path.join(imgpth, imageName)
    url2img(url=imageUrl,file=imgfile)
    annfile = os.path.join(annpth,markerName)
    json2txt(object,annfile)



# 瓦片地图
def loadgeoImageAndLabel(datajson,annpth='',imgpth=''):
    imageName = datajson['image_name'].split('.')[0]+'.jpg'
    datajson['image_name'] = imageName
    imageUrl = datajson['url']
    annName = datajson['image_name'].split('.')[0]+'.txt'


    if datajson['object']:

        objectlist = [ item for item in datajson['object'] if item['label'] == 'tree']


        if objectlist:
            datajson['object'] = objectlist

            imgfile = os.path.join(imgpth, imageName)
            url2img(url=imageUrl, file=imgfile)
            annfile = os.path.join(annpth, annName)
            json2polytxt(datajson, annfile)





if __name__ == '__main__':
    pth='/home/jasonwang/Documents/worktable/XAG/dataset/wire/电线分割标注第二批-V1已标注数据-2023年05月26日15时38分46秒'
    annpth='/home/jasonwang/Documents/worktable/XAG/dataset/wire/Annotations'
    imgpth='/home/jasonwang/Documents/worktable/XAG/dataset/wire/JPEGImages'

    if not os.path.exists(annpth):
        folder_check(annpth)

    if not os.path.exists(imgpth):
        folder_check(imgpth)

    # 瓦片
    # with open(pth,'r') as f:
    #     datajson=json.load(f)

    # for data in tqdm(datajson['data']):
    #     loadgeoImageAndLabel(data,annpth=annpth,imgpth=imgpth)


    # 图片
    filelist = os.listdir(pth)
    for file in tqdm(filelist):
        # print(file)
        filepath = os.path.join(pth,file)
        with open(filepath,'r') as f:
            datajson=json.load(f)
        print(datajson)
        loadLineImageAndLabel(datajson,annpth=annpth,imgpth=imgpth)
        # break






