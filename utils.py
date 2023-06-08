import cv2
import pandas as pd
import os
from tqdm import tqdm
import random
from shutil import copy2



def cutimg(input_path='',output_path='',filename='',size=(1024,1024)):
    imgpth=os.path.join(input_path,filename)
    img=cv2.imread(imgpth)
    h,w=img.shape[:2]

    for i in range(h//size[0]):
        for j in range(w//size[0]):
            item=img[i*size[0]:(i+1)*size[0],j*size[0]:(j+1)*size[0],:]
            file=os.path.join(output_path,filename[:-4]+'_'+str(i)+'_'+str(j)+'.png')
            print(file)
            cv2.imwrite(file,item)



def getimg(input_path,output_path):

    for name in tqdm(os.listdir(input_path)):
        if name[-4:]=='.JPG':
            filename = name[:-4]
            path=os.path.join(input_path,name)
            file=os.path.join(output_path,filename)

            if not os.path.exists(file):
                os.mkdir(file)

            cutimg(input_path=path,output_path=file,filename=name)





def randomImg(input_path,output_path,count=30):
    namelist=os.listdir(input_path)

    random.shuffle(namelist)
    for i,name in enumerate(namelist):
        if name[-4:] == '.png' :
            srcpath = os.path.join(input_path, name)
            dstpath = os.path.join(output_path,name)
            # cutimg(input_path=path, output_path=output_path, filename=name)
            copy2(srcpath,dstpath)
        if i>count:
            break







def cut_img(src,dst,y1,y2,x1,x2):
    img=cv2.imread(src)
    cimg=img[y1:y2,x1:x2,:]
    cv2.imwrite(dst,cimg)








if __name__ == '__main__':
    # input_path = './data/3米高度/5月11日'
    # output_path = './result/randomIMG/'
    # getimg(input_path,output_path)
    # randomImg(input_path,output_path)

    # path_list=os.listdir(output_path)
    # random.seed(5)
    # sample=random.sample(path_list,20)
    # print(sample)
    # for sap in sample:
    #     sapth=os.path.join(output_path,sap)
    #     savepth=os.path.join('./result/targetIMG',sap)
    #     img=cv2.imread(sapth)
    #     img=cv2.imwrite(savepth,img)


    #裁减核心区域
    # srcpth = '/home/jasonwang/Documents/worktable/XAG/dataset/riceseed/origal'
    # dstpth = '/home/jasonwang/Documents/worktable/XAG/dataset/riceseed/images'
    # namelist = os.listdir(srcpth)
    # for i,name in enumerate(namelist):
    #     print(name)
    #     src=os.path.join(srcpth,name)
    #     dst=os.path.join(dstpth,name)
    #
    #     cut_img(src=src,dst=dst,y1=500,y2=3148,x1=800,x2=4672)


    #裁切图片
    # srcpth = '/home/jasonwang/Documents/worktable/XAG/dataset/不同时期水稻穗/6'
    # dstpth = '/home/jasonwang/Documents/worktable/XAG/dataset/不同时期水稻穗/6_1'


    # namelist = os.listdir(srcpth)
    # for i,name in enumerate(namelist):
    #     cutimg(input_path=srcpth,output_path=dstpth,filename=name,size=(1024,1024))


    #随机获取图片
    srcpth = '/home/jasonwang/Documents/worktable/XAG/dataset/不同时期水稻穗/3_1'
    dstpth = '/home/jasonwang/Documents/worktable/XAG/dataset/不同时期水稻穗/3_2'
    # savepth = '/home/jasonwang/Documents/worktable/XAG/dataset/riceseed/JPGimages'
    randomImg(input_path=srcpth,output_path=dstpth,count=30)







