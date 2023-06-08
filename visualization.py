import cv2
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from shutil import copy2


def show_box(img,boxxes):
    for box in boxxes:
        cv2.rectangle(img, (box[0],box[1]), (box[2],box[3]), (0, 255, 0), 1)
    cv2.imshow('image',img)
    cv2.waitKey(0)


def show_minAreaRet(img,boxxes,savepth,colors=(255, 0, 0)):
    for box in boxxes:
        box = box.reshape(4,-1)
        cv2.drawContours(img, [box], 0, colors, 1)
        # cv2.polylines(img, [box], isClosed=True, color=(0, 255, 0), thickness=1)
    # cv2.imshow('image',img)
    cv2.imwrite(savepth,img)
    # cv2.waitKey(0)


def show_ploygon(img,boxxes,savepth,colors=(255, 0, 0)):
    # 创建绘图对象
    plt.figure(figsize=(547, 364))
    fig, ax = plt.subplots()
    # 显示原始图像
    ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    for box in boxxes:
        box=box.reshape(4,-1)
        # 创建多边形路径
        polygon = plt.Polygon(box, closed=True, edgecolor='g', linewidth=1, fill=None)
        # 添加多边形到绘图对象
        ax.add_patch(polygon)

    # 显示绘图结果
    # cv2.imwrite(savepth,img)
    plt.show()


if __name__ == '__main__':
    imgsrc = '/home/jasonwang/Documents/worktable/XAG/dataset/wire/dota_filter/train/images'
    annsrc =  '/home/jasonwang/Documents/worktable/XAG/dataset/wire/dota_filter/train/annfiles'
    visrc = '/home/jasonwang/Documents/worktable/XAG/dataset/wire/vis'




    imglist=os.listdir(imgsrc)
    annlist=os.listdir(annsrc)



    for i,ann in enumerate(annlist):
        name = ann.split('.')[0]
        print(name)
        imgpth=os.path.join(imgsrc,name+'.jpg')
        savepth=os.path.join(visrc,name+'.jpg')
        # anndstpth=os.path.join(anndst,name+'.txt')
        annpth=os.path.join(annsrc,name+'.txt')

        img = cv2.imread(imgpth)
        data = np.loadtxt(annpth,dtype='str').tolist()
        if data:
            ann=pd.read_csv(annpth,header=None,sep=' ')
            ann=ann.iloc[:,:8].values.astype('int')

            # show_minAreaRet(img,ann,savepth=savepth,colors=(255,0,0))
            show_minAreaRet(img,ann,savepth=savepth,colors=(255,0,0))
            # show_ploygon(img,ann,savepth=savepth,colors=(255,0,0))

            # #复制标签
            # copy2(annpth,anndstpth)
            # #复制图片
            # copy2(imgpth,savepth)
        else:
            # cv2.imshow('image',img)
            # cv2.waitKey(0)
            pass

        # break





