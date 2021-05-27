# -*- encoding: utf-8 -*-
from model import BiSeNet

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
from utils import *


def vis_parsing_maps(h, w, im, parsing_anno, stride, save_path, save_im=False):
    # Colors for all 20 parts
    part_colors = [[0, 0, 0], [255, 0, 0], [150, 30, 150], [255, 65, 255],
                   [150, 80, 0], [170, 120, 65], [220, 180, 210], [255, 125, 125],
                   [200, 100, 100], [215, 175, 125], [125, 125, 125], [255, 150, 0],
                   [255, 255, 0], [0, 255, 255], [255, 225, 120],  [125, 125, 255],
                   [0, 255, 0], [0, 0, 255],  [0, 150, 80]
                   ]

    im = np.array(im)
    vis_im = im.copy().astype(np.uint8)
    vis_parsing_anno = parsing_anno.copy().astype(np.uint8)
    vis_parsing_anno[vis_parsing_anno == 18] = 0   # hat
    # vis_parsing_anno[vis_parsing_anno == 17] = 0
    # vis_parsing_anno[vis_parsing_anno == 14] = 0  # neck
    # vis_parsing_anno[vis_parsing_anno == 15] = 0
    # vis_parsing_anno[vis_parsing_anno == 16] = 0  # cloth
    # vis_parsing_anno[vis_parsing_anno == 4] = 0

    vis_parsing_anno = cv2.resize(vis_parsing_anno, None, fx=stride, fy=stride, interpolation=cv2.INTER_NEAREST)
    vis_parsing_anno_color = np.zeros((vis_parsing_anno.shape[0], vis_parsing_anno.shape[1], 3)) + 255

    num_of_class = np.max(vis_parsing_anno)

    # 新增的一段，用于纠正错误的左右眉毛和眼睛
    index_nose = np.where(vis_parsing_anno == 10)
    index_lefteb = np.where(vis_parsing_anno == 2)
    index_righteb = np.where(vis_parsing_anno == 3)
    index_lefteye = np.where(vis_parsing_anno == 4)
    index_righteye = np.where(vis_parsing_anno == 5)
    index_leftear = np.where(vis_parsing_anno == 7)
    index_rightear = np.where(vis_parsing_anno == 8)

    nose_x = np.mean(index_nose[1])
    if index_lefteb:
        ind_false = np.where(index_lefteb[1] < nose_x)
        if ind_false:
            vis_parsing_anno[index_lefteb[0][ind_false], index_lefteb[1][ind_false]] = 3

    if index_righteb:
        ind_false = np.where(index_righteb[1] > nose_x)
        if ind_false:
            vis_parsing_anno[index_righteb[0][ind_false], index_righteb[1][ind_false]] = 2

    if index_lefteye:
        ind_false = np.where(index_lefteye[1] < nose_x)
        if ind_false:
            vis_parsing_anno[index_lefteye[0][ind_false], index_lefteye[1][ind_false]] = 5

    if index_righteye:
        ind_false = np.where(index_righteye[1] > nose_x)
        if ind_false:
            vis_parsing_anno[index_righteye[0][ind_false], index_righteye[1][ind_false]] = 4

    if index_leftear:
        ind_false = np.where(index_leftear[1] < nose_x)
        if ind_false:
            vis_parsing_anno[index_leftear[0][ind_false], index_leftear[1][ind_false]] = 8

    if index_rightear:
        ind_false = np.where(index_rightear[1] > nose_x)
        if ind_false:
            vis_parsing_anno[index_rightear[0][ind_false], index_rightear[1][ind_false]] = 7

    for pi in range(0, num_of_class+1):
        index = np.where(vis_parsing_anno == pi)
        vis_parsing_anno_color[index[0], index[1], :] = part_colors[pi]

    vis_parsing_anno_color = vis_parsing_anno_color.astype(np.uint8)
    # print(vis_parsing_anno_color.shape, vis_im.shape)
    # vis_im = cv2.addWeighted(cv2.cvtColor(vis_im, cv2.COLOR_RGB2BGR), 0.0, vis_parsing_anno_color, 1.0, 0)

    vis_im = vis_parsing_anno_color
    # Save result or not
    if save_im:
        # 这里的save_path要注意保存成png，如果是jpg，则需要将cv2.IMWRITE_JPEG_QUALITY设置为100，否则语义标签值在保存的时候会损失
        vis_im = cv2.resize(vis_im, (w, h))
        cv2.imwrite(save_path, vis_im[:, :, ::-1], [int(cv2.IMWRITE_JPEG_QUALITY), 100])

    return vis_im


# 不带pose版本
def evaluate(respth='', dspth='', croppth="", cp='79999_iter.pth'):

    if not os.path.exists(respth):
        os.makedirs(respth)

    image_list = sorted(glob.glob(dspth + "*.png")) 

    n_classes = 19
    net = BiSeNet(n_classes=n_classes)
    net.cuda()
    save_pth = osp.join('res/cp', cp)
    net.load_state_dict(torch.load(save_pth))
    net.eval()

    to_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    with torch.no_grad():
        for i in range(0, len(image_list)):   
            image_path = image_list[i]
            # subdir = image_path.split("/")[-2]
            name = image_path.split("/")[-1]

            save_dir = respth
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

            img = Image.open(image_path)
            h, w, c = np.array(img).shape
            image = img.resize((512, 512), Image.BILINEAR)

            # 先进行normalize,调节成和SEAN输入图像类似的人脸比例
            temp = normalize_SEAN(np.array(image), 0)
            image = Image.fromarray(temp)

            # 为了测试侧脸图像加入的平移
            # image = image.rotate(0, expand=1, fillcolor=(255, 255, 255), translate=(-80, 0))

            img = to_tensor(image)
            img = torch.unsqueeze(img, 0)
            img = img.cuda()
            out = net(img)[0]
            parsing = out.squeeze(0).cpu().numpy().argmax(0)

            name2 = name

            if not os.path.exists(respth):   
                os.makedirs(respth)         
            vis_im = vis_parsing_maps(h, w, image, parsing, stride=1, save_path=osp.join(respth, name2), save_im=False)   
            print(i, image_path)
            Image.fromarray(vis_im).save(osp.join(respth, name2))        


            if croppth != "":
                if not os.path.exists(croppth):
                    os.makedirs(croppth)

                crop, full_color, ret = get_stylegan_parsing_pair(np.array(image), vis_im)
                if ret > -1:
                    Image.fromarray(crop).save(osp.join(croppth, name2))   


def draw_landmark(img, landmark_arr):
    # 只绘制外轮廓的点，并连线
    point_size = 2
    point_color = (0, 255, 0)
    thickness = 4
    r, c = landmark_arr.shape
    if r > c:
        for i in range(0, 27):
            x = landmark_arr[i, 0]
            y = landmark_arr[i, 1]
            cv2.circle(img, (int(x), int(y)), point_size, point_color, thickness)
            cv2.putText(img, str(i), (int(x) + 1, int(y) + 1), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 0, 0), 1)
            landmark2 = np.zeros((27, 2))
            for i in range(0, 17):
                landmark2[i, 0] = landmark_arr[i, 0]
                landmark2[i, 1] = landmark_arr[i, 1]

            ind = 17
            for i in range(26, 16, -1):
                landmark2[ind, 0] = landmark_arr[i, 0]
                landmark2[ind, 1] = landmark_arr[i, 1]
                ind += 1

            # landmark2 = np.expand_dims(landmark2, axis=1)
            landmark2 = landmark2.reshape((-1, 1, 2)).astype(np.int32)

            contours = [landmark2]
            ind = 0
            mask = np.zeros_like(img)
            cv2.drawContours(mask, contours, ind, (255, 255, 255), -1)
            cv2.drawContours(img, contours, ind, (255, 255, 0), 0)
    elif r < c:
        for i in range(0, 27):
            x = landmark_arr[0, i]
            y = landmark_arr[1, i]
            cv2.circle(img, (int(x), int(y)), point_size, point_color, thickness)
            cv2.putText(img, str(i), (int(x) + 1, int(y) + 1), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 0, 0), 1)
            mask = np.zeros_like(img)
    return img, mask

def get_faceparsing_partial(img):
    landmark, ret = get_landmark_from_img(img)
    if ret < 0:
        return img, ret
    mask = draw_landmark_contour(img, landmark)
    # mask2 = np.expand_dims(mask, axis=2)
    # mask2 = np.concatenate((mask2, mask2, mask2), axis=2)/255
    img = img*np.uint8(mask/255)

    img[np.where(mask == 0)] = 255

    return img, ret


def align_crop_full(color_crop, color_full):
    # 对齐crop和full的语义分割图-【旧版本】
    # 新版本函数是get_stylegan_parsing_pair
    label_crop = parsing_label2celeba(parsing_Color2label(color_crop))
    label_full = parsing_label2celeba(parsing_Color2label(color_full))
    index1 = np.where(label_crop == 2)
    if (not index1) or len(index1[0]) == 0:
        print("no nose")
        return color_crop

    row = np.min(index1[0])

    index_face_full = np.where(label_full == 1)
    index = ([i for i in index_face_full[0] if i>row], [index_face_full[1][j] for j in range(0, len(index_face_full[1])) if index_face_full[0][j]>row])
    color_crop[index[0], index[1], :] = color_full[index[0], index[1], :]

    return color_crop

def normalize_SEAN(img, flag=1):
    # 相当于把图像放大1.1倍后再向下移动２０像素
    scale = 1.1
    if flag:
        img = cv2.resize(img, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
    else:
        img = cv2.resize(img, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
    res = []

    if len(img.shape) == 2:
        res = np.zeros((512, 512), dtype=np.uint8)
        left = img.shape[0] // 2 - 256
        top = max(0, img.shape[0] // 2 - 256 - 20)
        res[:, :] = img[top:top + 512, left:left + 512]

    elif len(img.shape) == 3 and img.shape[2] == 3:
        res = np.ones((512, 512, 3), dtype=np.uint8) * 255
        left = img.shape[0] // 2 - 256
        top = max(0, img.shape[0] // 2 - 256 - 20)
        res[:, :, :] = img[top:top + 512, left:left + 512, :]

    return res


if __name__ == "__main__":
    # 0324
    dspth = "/home/zhang/zydDataset/faceRendererData/testResults/0_facerenderer-pix2pix/0324/"
    respth = "/home/zhang/zydDataset/faceRendererData/testResults/1_face-parsing/0324/full/"
    croppth = "/home/zhang/zydDataset/faceRendererData/testResults/1_face-parsing/0324/crop/"

    evaluate(respth, dspth, croppth)


# 0311部分单独备份，涉及到多个pose的转换
# def evaluate(respth='', dspth='', croppth="", cp='79999_iter.pth'):
#
#     if not os.path.exists(respth):
#         os.makedirs(respth)
#
#     image_list = sorted(glob.glob(dspth + "*/*.png"))  # 0311
#
#     n_classes = 19
#     net = BiSeNet(n_classes=n_classes)
#     net.cuda()
#     save_pth = osp.join('res/cp', cp)
#     net.load_state_dict(torch.load(save_pth))
#     net.eval()
#
#     to_tensor = transforms.Compose([
#         transforms.ToTensor(),
#         transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
#     ])
#     with torch.no_grad():
#         for i in range(0, len(image_list)):   # len(image_list)
#             image_path = image_list[i]
#             # subdir = image_path.split("/")[-2]
#             name = image_path.split("/")[-1]
#             pose_ind = image_path.split("/")[-2]  # 0311
#
#             save_dir = respth
#             if not os.path.exists(save_dir):
#                 os.makedirs(save_dir)
#
#             img = Image.open(image_path)
#             h, w, c = np.array(img).shape
#             # cv2.imwrite("./img.png", np.array(img))
#
#             image = img.resize((512, 512), Image.BILINEAR)
#
#             # 试试先进行normalize
#             temp = normalize_SEAN(np.array(image), 0)
#             image = Image.fromarray(temp)
#             image = image.rotate(0, expand=1, fillcolor=(0, 0, 0), translate=(-10*int(pose_ind), 0))   # 0311
#
#             # 为了测试侧脸图像加入的
#             # image = image.rotate(0, expand=1, fillcolor=(255, 255, 255), translate=(-80, 0))
#
#             img = to_tensor(image)
#             img = torch.unsqueeze(img, 0)
#             img = img.cuda()
#             out = net(img)[0]
#             parsing = out.squeeze(0).cpu().numpy().argmax(0)
#
#             name2 = name
#             # =========================================================================================================
#             respth2 = respth + pose_ind + "/"      # 0311
#
#             if not os.path.exists(respth2):   # 0311
#                 os.makedirs(respth2)          # 0311
#             vis_im = vis_parsing_maps(h, w, image, parsing, stride=1, save_path=osp.join(respth2, name2),
#                                       save_im=False)  # 0311
#             print(i, image_path)
#             Image.fromarray(vis_im).save(osp.join(respth2, name2))
#             croppth2 = croppth + pose_ind + "/"
#
#             if croppth2 != "":
#                 if not os.path.exists(croppth2):
#                     os.makedirs(croppth2)
#
#                 crop, full_color, ret = get_stylegan_parsing_pair(np.array(image), vis_im)
#                 if ret > -1:
#                     Image.fromarray(crop).save(osp.join(croppth2, name2))
#
#             # if croppth != "":
#             #     if not os.path.exists(croppth):
#             #         os.makedirs(croppth)
#             #
#             #     crop, full_color, ret = get_stylegan_parsing_pair(np.array(image), vis_im)   # 这里已经对齐了
#             #     if ret > -1:
#             #         Image.fromarray(crop).save(osp.join(croppth, name2))
#
#
# if __name__ == "__main__":
#     # 0311
#     dspth = "/home/zhang/zydDataset/faceRendererData/testResults/0_facerenderer-pix2pix/0325/"
#     respth = "/home/zhang/zydDataset/faceRendererData/testResults/1_face-parsing/0325/full/"
#     croppth = "/home/zhang/zydDataset/faceRendererData/testResults/1_face-parsing/0325/crop/"
#     evaluate(respth, dspth, croppth)

