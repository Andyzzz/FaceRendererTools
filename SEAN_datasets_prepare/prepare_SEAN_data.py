# -*- encoding: utf-8 -*-
from model import BiSeNet
import sys


import torch

import os
import os.path as osp
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import cv2
import csv
import time
import glob
from process_data_asian import draw_landmark_contour, get_landmark_from_img
import face_alignment
# import matplotlib.pyplot as plt
from utils import *


def refine_landmark_using_parsing(landmark, label_parsing):
    for i in range(0, 17):
        row = int(landmark[i, 0])
        col = int(landmark[i, 1])
        index = np.where(label_parsing[:, col] == 1)
        if not np.any(index[0]):
            index = np.where(label_parsing[row, :] == 1)
            if not index:
                continue

            left = np.min(index[0])
            right = np.max(index[0])
            if abs(col - left) < abs(col - right):
                landmark[i, 1] = left
            else:
                landmark[i, 1] = right

        bottom = np.max(index[0])
        landmark[i, 0] = bottom


    return landmark



def get_celeba_parsing_pair(image, label_celeba):
    label_parsing = celeba_label2parsinglabel(label_celeba)
    color = parsing_label2color(label_parsing)
    landmark, ret = get_landmark_from_img(image)

    # if ret < 0:
    #     return None, ret

    # refine dlib landmark
    # landmark = refine_landmark_using_parsing(landmark, label_parsing)

    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False, face_detector='dlib')

    preds = fa.get_landmarks(image)
    landmark = preds[0]

    mask = draw_landmark_contour(image, landmark)
    crop = color * np.uint8(mask / 255)

    return crop, color, 0


def celeba_dataset():
    # 用celeba制作语义分割补全的训练集  crop-->full
    lis = sorted(glob.glob("/run/user/1000/gvfs/afp-volume:host=CITE-3D.local,user=anonymous,volume=share/zhangyidi/CelebA-HQ-img/*.jpg"))
    label_dir = "/run/user/1000/gvfs/afp-volume:host=CITE-3D.local,user=anonymous,volume=share/zhangyidi/CelebAMask-HQ-mask/"
    tgt_full_color_dir = "/run/user/1000/gvfs/afp-volume:host=CITE-3D.local,user=anonymous,volume=share/zhangyidi/celeba_full_parsing/"
    tgt_crop_dir = "/run/user/1000/gvfs/afp-volume:host=CITE-3D.local,user=anonymous,volume=share/zhangyidi/celeba_crop/"
    for i in range(0, 10):
        name = str(i) + ".jpg"
        imgname = "/run/user/1000/gvfs/afp-volume:host=CITE-3D.local,user=anonymous,volume=share/zhangyidi/CelebA-HQ-img/" + name
        # name = lis[i].split("/")[-1]
        name = name.replace("jpg", "png")
        img = np.array(Image.open(lis[i]).resize((512, 512)))
        label = np.array(Image.open(label_dir + name))
        crop, full_color, ret = get_celeba_parsing_pair(img, label)
        if ret < 0:
            continue

        Image.fromarray(crop).save(tgt_crop_dir + name)
        Image.fromarray(full_color).save(tgt_full_color_dir + name)
        print(i)

# 生成SEAN可以测试的数据集
def get_facescape_label():
    image_dir = "/run/user/1000/gvfs/smb-share:server=cite-3d.local,share=share/zhangyidi/FaceRendererData/testResults/2_facerender-pix2pix-hair/0517_addhair/"
    save_path = "/run/user/1000/gvfs/smb-share:server=cite-3d.local,share=share/zhangyidi/FaceRendererData/testResults/3_SEAN/datasets/0517_addhair/test_label2/"
    rgbimg_path = "/run/user/1000/gvfs/smb-share:server=cite-3d.local,share=share/zhangyidi/FaceRendererData/testResults/0_facerenderer-pix2pix/0517/"
    img_path = "/run/user/1000/gvfs/smb-share:server=cite-3d.local,share=share/zhangyidi/FaceRendererData/testResults/3_SEAN/datasets/0517_addhair/test_img/"

    lis = sorted(glob.glob(image_dir + "*.png"))
    for i in range(0, len(lis)):
        # subdir = lis[i].split("/")[-2]
        name = lis[i].split("/")[-1]
        src_image_path = lis[i]
        src_parsing = np.array(Image.open(src_image_path))
        src_label = parsing_Color2label(src_parsing)
        celeba_label = parsing_label2celeba(src_label)

        # ==================== temp test ===================
        celeba_label[celeba_label == 8] = 9
        # ==================================================

        # 这里先存储到test_label2而不是直接存储到test_label里的原因是，原来的模型输出结果会在脸部多出来一圈，需要使用郭的代码与crop重新对齐
        #　facerenderer-pix2pix-hair的checkpoint_align里的模型没有这个问题了，可以直接保存到test_label里
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        save_name = os.path.join(save_path, name)

        Image.fromarray(celeba_label).save(save_name)

        rgb_image = cv2.imread(rgbimg_path + name)[:,:, ::-1]

        rgb_image = normalize_SEAN(rgb_image)
        rgb_image = Image.fromarray(rgb_image)
        # rgb_image = rgb_image.rotate(0, expand=1, fillcolor=(255, 255, 255), translate=(-60, 0))

        if not os.path.exists(img_path):
            os.makedirs(img_path)
        rgb_image.save(save_name.replace("test_label2", "test_img"))
        print(i)


# 生成SEAN可以测试的数据集
def get_facescape_label_0311():
    image_dir = "/home/zhang/zydDataset/faceRendererData/testResults/2_facerender-pix2pix-hair/0325/"
    lis = sorted(glob.glob(image_dir + "*/*.png"))   # 0311
    for i in range(0, len(lis)):
        name = lis[i].split("/")[-1]
        pose_ind = lis[i].split("/")[-2]   # pose_ind  0311
        src_image_path = lis[i]
        src_parsing = np.array(Image.open(src_image_path))
        src_label = parsing_Color2label(src_parsing)
        celeba_label = parsing_label2celeba(src_label)

        # ==================== temp test ===================
        # celeba_label[celeba_label == 8] = 13
        # celeba_label[celeba_label == 9] = 13
        # ==================================================
        celeba_label[celeba_label == 8] = 9
        save_path = "/home/zhang/PycharmProjects/SEAN/datasets/facescape_0325/" + pose_ind + "/test_label/"   # 0311
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        # save_name = os.path.join(save_path, subdir + "_" + name)
        save_name = os.path.join(save_path, name)

        Image.fromarray(celeba_label).save(save_name)

        rgb_image = cv2.imread(
            "/home/zhang/zydDataset/faceRendererData/testResults/0_facerenderer-pix2pix/0325/" + pose_ind + "/" + name)[:,
                    :, ::-1]   # 0311
        rgb_image = normalize_SEAN(rgb_image)
        rgb_image = Image.fromarray(rgb_image)
        rgb_image = rgb_image.rotate(0, expand=1, fillcolor=(255, 255, 255), translate=(-10*int(pose_ind), 0))   # 0311

        img_path = "/home/zhang/PycharmProjects/SEAN/datasets/facescape_0325/" + pose_ind + "/test_img/"
        if not os.path.exists(img_path):
            os.makedirs(img_path)

        rgb_image.save(save_name.replace("test_label", "test_img"))
        print(i)



if __name__ == "__main__":

    get_facescape_label()


