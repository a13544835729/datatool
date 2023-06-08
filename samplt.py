import numpy as np
import pandas as pd
import os
from shutil import copy2




def filter_data(img_src_pth,ann_src_pth,img_dst_pth,ann_dst_pth):
    for i,filename in enumerate(os.listdir(ann_src_pth)):
        ann_file_path = os.path.join(ann_src_pth,filename)
        ann_save_path = os.path.join(ann_dst_pth,filename)
        img_file_path = os.path.join(img_src_pth,filename[:-4]+'.jpg')
        img_save_path = os.path.join(img_dst_pth,filename[:-4]+'.jpg')

        # print(ann_file_path)
        try:
            ann = pd.read_csv(ann_file_path, header=None, sep=' ')
            copy2(ann_file_path,ann_save_path)
            copy2(img_file_path,img_save_path)
            # print(ann)
        except Exception as e:
            print(e)




if __name__ == '__main__':
    img_src_pth = '/home/jasonwang/Documents/worktable/XAG/dataset/wire/dota_split/val/images'
    ann_src_pth = '/home/jasonwang/Documents/worktable/XAG/dataset/wire/dota_split/val/annfiles'
    img_dst_pth = '/home/jasonwang/Documents/worktable/XAG/dataset/wire/dota_filter/val/images'
    ann_dst_pth = '/home/jasonwang/Documents/worktable/XAG/dataset/wire/dota_filter/val/annfiles'
    filter_data(img_src_pth,ann_src_pth,img_dst_pth,ann_dst_pth)