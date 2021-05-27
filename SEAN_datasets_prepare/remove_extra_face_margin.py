import numpy as np
import glob
from PIL import Image
import cv2
import tqdm
import os
from utils import *


##首先把crop的rgb图片转换成normalized图片，然后转成单通道label存储下来，并存储相应的rgb图片方便可视化。
list_crop = glob.glob("/run/user/1000/gvfs/smb-share:server=cite-3d.local,share=share/zhangyidi/FaceRendererData/testResults/1_face-parsing/0517/crop/*.png")#crop face的parsing彩色图像
list_crop = sorted(list_crop)

list_pred = glob.glob("/run/user/1000/gvfs/smb-share:server=cite-3d.local,share=share/zhangyidi/FaceRendererData/testResults/3_SEAN/datasets/0517_addhair/test_label2/*.png")#整个面部的单通道label图像路径
list_pred = sorted(list_pred)
save_path = "/run/user/1000/gvfs/smb-share:server=cite-3d.local,share=share/zhangyidi/FaceRendererData/testResults/3_SEAN/datasets/0517_addhair/test_label/"#存储路径

if not os.path.exists(save_path):
    os.makedirs(save_path)

for i in tqdm.trange(len(list_crop)):
# for i in tqdm.trange(40):
    test_crop = Image.open(list_crop[i])
    test_pred = Image.open(list_pred[i])
    crop_arr = np.array(test_crop)
    pred_arr = np.array(test_pred)

    # crop_arr = normalize_SEAN(parsing_label2celeba(parsing_Color2label(crop_arr)))    # add by zyd
    crop_arr = parsing_label2celeba(parsing_Color2label(crop_arr))    # modified by zyd on 2021.3.9


    src_sub_arr = pred_arr - crop_arr
    # cv2.imwrite("/home/zhang/zydDataset/faceRendererData/testResults/src_sub_arr.png", src_sub_arr*10)
    # cv2.imshow("img", crop_arr*20)
    # cv2.waitKey()
    src_sub_arr_rgb = celeba_label2color(src_sub_arr)
    src_sub_rgb_img = Image.fromarray(src_sub_arr_rgb)
    name1 = list_crop[i].split("/")[-1]
    # src_sub_rgb_img.save(save_path + "src_sub_rgb_img/" + name1)  ##存储原始的没有补全的相减的差值rgb image

    L_eye_mask = (pred_arr == 4)
    R_eye_mask = (pred_arr == 5)
    eye_row = np.where(L_eye_mask==1)[0]
    eye_col = np.where(L_eye_mask==1)[1]
    eye_row_mean = 0
    eye_col_mean = 0
    if eye_row.size > 200:   #200为眼睛像素阈值，手动设定。（认为小于200的不是眼睛，label预测错误，下面用眉毛代替）
        eye_row_mean = sum(eye_row)/max(1, eye_row.size)   #求出左眼睛的均值位置
        eye_col_mean = sum(eye_col)/max(1, eye_col.size)

    R_eye_row = np.where(R_eye_mask==1)[0]
    R_eye_col = np.where(R_eye_mask==1)[1]
    R_eye_row_mean = 0
    R_eye_col_mean = 0
    if R_eye_row.size > 200:
        R_eye_row_mean = sum(R_eye_row)/max(1, R_eye_row.size)  #求出右眼睛的均值位置
        R_eye_col_mean = sum(R_eye_col)/max(1, R_eye_col.size)
    
    row_eye_max = max(int(R_eye_row_mean),int(eye_row_mean))   #两眼之中较高的row值用于分割额头不变部分和额头以下用于填充部分
    if row_eye_max == 0:  #如果没有眼睛，则按眉毛的地方划分上下区域
        L_brow_mask = (pred_arr == 6)
        R_brow_mask = (pred_arr == 7)
        L_brow_row = np.where(L_brow_mask==1)[0]
        L_brow_col = np.where(L_brow_mask==1)[1]
        L_brow_row_mean = sum(L_brow_row)/max(1, L_brow_row.size)   #求出右眼睛的均值位置
        L_brow_col_mean = sum(L_brow_col)/max(1, L_brow_col.size)

        R_brow_row = np.where(R_brow_mask==1)[0]
        R_brow_col = np.where(R_brow_mask==1)[1]
        R_brow_row_mean = sum(R_brow_row)/max(1, R_brow_row.size)#求出右眼睛的均值位置
        R_brow_col_mean = sum(R_brow_col)/max(1, R_brow_col.size)
        
        row_brow_max = max(int(R_brow_row_mean),int(L_brow_row_mean))
        row_eye_max = row_brow_max + 50#眉毛位置+50用于接近眼睛位置，和上面用眼睛位置的尽可能保持一致
    if row_eye_max>260:#如果最终算出的上下分割的值大于260，说明算的太大了不合理，直接置为260
        row_eye_max = 260
    
    sub_red = src_sub_arr
    sub_red_mask = (sub_red == 1)  #剩余的所有skin区域
    sub_red_mask_img = Image.fromarray(np.array(sub_red_mask * 255, dtype = np.uint8))
    # sub_red_mask_img.save(save_path + "mask/sub_red_mask_img/" + name1)
    eye_dot_mask = np.zeros((512,512),dtype = np.uint8)
    eye_dot_mask[row_eye_max:,:] = 1   #眼睛以下的skin区域
    eye_red_mask = sub_red_mask * eye_dot_mask
    eye_red_mask_img = Image.fromarray(np.array(eye_red_mask * 255, dtype = np.uint8))   ##eye_red_mask_img中间mask查看
    # eye_red_mask_img.save(save_path + "mask/eye_red_mask_img/" + name1)

    neck = (pred_arr == 17)
    # ============================ zyd add =========================================
    if np.all(neck == False):
        neck = (pred_arr == 18)
    # ==============================================================================
    neck_img = Image.fromarray(np.array(neck * 255, dtype=np.uint8))
    # neck_img.save(save_path + "mask/neck_img/" + name1)

    neck_col = np.where(neck==1)[1]
    neck_col_min = neck_col.min()   #计算脖子的左右最大值
    neck_col_max = neck_col.max()
    neck_dot_mask = np.zeros((512,512), dtype=np.uint8)
    neck_dot_mask[:,neck_col_min:neck_col_max] = 1
    neck_red_mask = eye_red_mask * neck_dot_mask   #眼睛以下的，脖子以上的skin区域
    neck_red_mask_img = Image.fromarray(np.array(neck_red_mask * 255, dtype = np.uint8))
    # neck_red_mask_img.save(save_path + "mask/neck_red_mask_img/" + name1)

    cheek_extra_mask = eye_red_mask - neck_red_mask

    ####获取完mask，进行填充语义
    pred_new = pred_arr
    neck_fill_mask = np.ones((512,512)) * neck_red_mask * 17
    
    
    pred_new_fill_label = pred_new

    cheek_extra_index = np.where(cheek_extra_mask == 1)#除 眼睛以下的，脖子以上的skin区域 外的区域，先填充此部分，再填充脖子以上的，防止水平横线分割的情况
    for i in range(cheek_extra_index[0].size):
        j = 0
        row = cheek_extra_index[0][i]##要补充点的row和col
        col = cheek_extra_index[1][i]
        while(j<100):
            j = j+1
            if(col-j<0 or col+j>511):#搜索越界
                break
            temp_label_l = pred_new_fill_label[row,col-j]#左右横向搜索非skin区域
            temp_label_r = pred_new_fill_label[row,col+j]
            if(temp_label_l != 1):
                pred_new_fill_label[row,col] = temp_label_l
                break
            elif(temp_label_r != 1):
                pred_new_fill_label[row,col] = temp_label_r
                break
            else:
                pass
            
    pred_new_fill_label = pred_new_fill_label * (1-neck_red_mask) + neck_fill_mask ##填充脖子
    
    pred_new_fill_label_img = Image.fromarray(np.array(pred_new_fill_label,dtype=np.uint8))#存储label图像
    pred_new_fill_label_img.save(save_path + name1)
    print(name1)


