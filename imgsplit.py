import os
import random
from shutil import copy2


def data_set_split(src_data_folder, target_data_folder, slice_data=[0.7, 0.2, 0.1]):
    '''
    读取源数据文件夹，生成划分好的文件夹，分为trian、val、test三个文件夹进行
    :param src_data_folder: r"D:\Desktop\segmentation_2021\data"
    :param target_data_folder: 目标文件夹 r"D:\Desktop\segmentation_2021\a"
    :param slice_data: 划分数据比例比例  训练 验证 测试所占百分比
    :return:
    '''
    print("开始数据集划分")
    # class_names = os.listdir(src_data_folder)
    # 在目标目录下创建文件夹
    img_target_path = os.path.join(target_data_folder, 'images')
    label_target_path = os.path.join(target_data_folder, 'labels')
    if not os.path.exists(img_target_path):
        os.mkdir(img_target_path)
    if not os.path.exists(label_target_path):
        os.mkdir(label_target_path)


    train_img_target_path=os.path.join(img_target_path, 'train')
    if not os.path.exists(train_img_target_path):
        os.mkdir(train_img_target_path)

    val_img_target_path=os.path.join(img_target_path, 'val')
    if not os.path.exists(val_img_target_path):
        os.mkdir(val_img_target_path)

    test_img_target_path=os.path.join(img_target_path, 'test')
    if not os.path.exists(test_img_target_path):
        os.mkdir(test_img_target_path)


    train_label_target_path = os.path.join(label_target_path, 'train')
    if not os.path.exists(train_label_target_path):
        os.mkdir(train_label_target_path)

    val_label_target_path = os.path.join(label_target_path, 'val')
    if not os.path.exists(val_label_target_path):
        os.mkdir(val_label_target_path)

    test_label_target_path = os.path.join(label_target_path, 'test')
    if not os.path.exists(test_label_target_path):
        os.mkdir(test_label_target_path)


    # img_split_path
    current_data_path = os.path.join(src_data_folder,'images')
    current_label_path= os.path.join(src_data_folder,'annfiles')
    current_all_data = os.listdir(current_data_path)
    current_data_length = len(current_all_data)
    current_data_index_list = list(range(current_data_length))
    random.shuffle(current_data_index_list)


    train_stop_flag = current_data_length * slice_data[0]
    val_stop_flag = current_data_length * (slice_data[0] + slice_data[1])
    current_idx = 0
    train_num = 0
    val_num = 0
    test_num = 0
    for i in current_data_index_list:
        labelname=current_all_data[i].split('.')[0]
        src_img_path = os.path.join(current_data_path, current_all_data[i])
        src_label_path= os.path.join(current_label_path,labelname+'.txt')

        if current_idx <= train_stop_flag:
            copy2(src_img_path, train_img_target_path)
            copy2(src_label_path, train_label_target_path)
            train_num = train_num + 1

        elif (current_idx > train_stop_flag) and (current_idx <= val_stop_flag):
            copy2(src_img_path, val_img_target_path)
            copy2(src_label_path, val_label_target_path)
            val_num = val_num + 1
        else:
            copy2(src_img_path, test_img_target_path)
            copy2(src_label_path, test_label_target_path)
            # print("{}复制到了{}".format(src_img_path, test_folder))
            test_num = test_num + 1

        current_idx = current_idx + 1


        print("训练集{}张".format(train_num))
        print("验证集{}张".format(val_num))
        print("测试集{}张".format(test_num))



def dataset_split(src_data_folder, target_data_folder, slice_data=[0.7, 0.2, 0.1]):
    '''
    读取源数据文件夹，生成划分好的文件夹，分为trian、val、test三个文件夹进行
    :param src_data_folder: r"D:\Desktop\segmentation_2021\data"
    :param target_data_folder: 目标文件夹 r"D:\Desktop\segmentation_2021\a"
    :param slice_data: 划分数据比例比例  训练 验证 测试所占百分比
    :return:
    '''
    print("开始数据集划分")


    train_target_path=os.path.join(target_data_folder, 'train')
    if not os.path.exists(train_target_path):
        os.mkdir(train_target_path)

    val_target_path=os.path.join(target_data_folder, 'val')
    if not os.path.exists(val_target_path):
        os.mkdir(val_target_path)

    test_target_path=os.path.join(target_data_folder, 'test')
    if not os.path.exists(test_target_path):
        os.mkdir(test_target_path)

    # train
    train_img_target_path = os.path.join(train_target_path, 'images')
    train_label_target_path = os.path.join(train_target_path, 'annfiles')
    if not os.path.exists(train_img_target_path):
        os.mkdir(train_img_target_path)
    if not os.path.exists(train_label_target_path):
        os.mkdir(train_label_target_path)

    # val
    val_img_target_path = os.path.join(val_target_path, 'images')
    val_label_target_path = os.path.join(val_target_path, 'annfiles')
    if not os.path.exists(val_img_target_path):
        os.mkdir(val_img_target_path)
    if not os.path.exists(val_label_target_path):
        os.mkdir(val_label_target_path)


    # test
    test_img_target_path = os.path.join(test_target_path, 'images')
    test_label_target_path = os.path.join(test_target_path, 'annfiles')
    if not os.path.exists(test_img_target_path):
        os.mkdir(test_img_target_path)
    if not os.path.exists(test_label_target_path):
        os.mkdir(test_label_target_path)



    # img_split_path
    current_data_path = os.path.join(src_data_folder,'images')
    current_label_path= os.path.join(src_data_folder,'annfiles')
    current_all_data = os.listdir(current_data_path)
    current_data_length = len(current_all_data)
    current_data_index_list = list(range(current_data_length))
    random.shuffle(current_data_index_list)


    train_stop_flag = current_data_length * slice_data[0]
    val_stop_flag = current_data_length * (slice_data[0] + slice_data[1])
    current_idx = 0
    train_num = 0
    val_num = 0
    test_num = 0
    for i in current_data_index_list:
        labelname=current_all_data[i].split('.')[0]
        src_img_path = os.path.join(current_data_path, current_all_data[i])
        src_label_path= os.path.join(current_label_path,labelname+'.txt')

        if current_idx <= train_stop_flag:
            copy2(src_img_path, train_img_target_path)
            copy2(src_label_path, train_label_target_path)
            train_num = train_num + 1

        elif (current_idx > train_stop_flag) and (current_idx <= val_stop_flag):
            copy2(src_img_path, val_img_target_path)
            copy2(src_label_path, val_label_target_path)
            val_num = val_num + 1
        else:
            copy2(src_img_path, test_img_target_path)
            copy2(src_label_path, test_label_target_path)
            # print("{}复制到了{}".format(src_img_path, test_folder))
            test_num = test_num + 1

        current_idx = current_idx + 1


        print("训练集{}张".format(train_num))
        print("验证集{}张".format(val_num))
        print("测试集{}张".format(test_num))


if __name__ == '__main__':
    src_data_folder = "/home/jasonwang/Documents/worktable/XAG/dataset/wire/original"
    target_data_folder = '/home/jasonwang/Documents/worktable/XAG/dataset/wire/dota'
    # data_set_split(src_data_folder, target_data_folder, slice_data=[0.7, 0.2, 0.1])
    dataset_split(src_data_folder, target_data_folder, slice_data=[0.7, 0.2, 0.1])
