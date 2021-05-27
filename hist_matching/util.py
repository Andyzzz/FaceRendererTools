import cv2  
import numpy as np  
import sys  
import glob
from get_pair_parsing2 import *
import os.path as osp


def parsing_Color2label(img):
    # convert the [face-parsing.Pytorch]-format RGB image to [face-parsing.Pytorch]-format labels (single channel).
    color_list = [[0, 0, 0], [255, 0, 0], [150, 30, 150], [255, 65, 255],
                  [150, 80, 0], [170, 120, 65], [220, 180, 210], [255, 125, 125],
                  [200, 100, 100], [215, 175, 125], [125, 125, 125], [255, 150, 0],
                  [255, 255, 0], [0, 255, 255], [255, 225, 120], [125, 125, 255],
                  [0, 255, 0], [0, 0, 255], [0, 150, 80]
                  ]

    label = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
    for i in range(0, len(color_list)):  # len(colors)
        color = color_list[i]
        index = np.where(np.all(img == color, axis=-1))
        label[index[0], index[1]] = i

    return label


def parsing_label2celeba(label):
    # convert the [face-parsing.Pytorch]-format label image to [CelebAMask-HQ]-format label image
    map_list = [0, 1, 6, 7, 4, 5, 3, 8, 9, 15, 2, 10, 11, 12, 17, 16, 18, 13, 14]
    res = label.copy()
    for i in range(0, len(map_list)):
        index = np.where(label == i)
        res[index[0], index[1]] = map_list[i]

    return res


def celeba_label2color(label):
    # convert the [CelebAMask-HQ]-format label image to [CelebAMask-HQ]-format RGB image
    color_list = [[0, 0, 0], [204, 0, 0], [76, 153, 0], [204, 204, 0],
                  [51, 51, 255], [204, 0, 204], [0, 255, 255], [255, 204, 204],
                  [102, 51, 0], [255, 0, 0], [102, 204, 0], [255, 255, 0],
                  [0, 0, 153], [0, 0, 204], [255, 51, 153], [0, 204, 204],
                  [0, 51, 0], [255, 153, 51], [0, 204, 0]]
    res = np.zeros((label.shape[0], label.shape[1], 3), dtype=np.uint8)
    for idx, color in enumerate(color_list):
        res[label == idx] = color

    return res


def celeba_color2label(img):
    # convert the [CelebAMask-HQ]-format RGB image to [CelebAMask-HQ]-format label image
    color_list = [[0, 0, 0], [204, 0, 0], [76, 153, 0], [204, 204, 0],
                  [51, 51, 255], [204, 0, 204], [0, 255, 255], [255, 204, 204],
                  [102, 51, 0], [255, 0, 0], [102, 204, 0], [255, 255, 0],
                  [0, 0, 153], [0, 0, 204], [255, 51, 153], [0, 204, 204],
                  [0, 51, 0], [255, 153, 51], [0, 204, 0]]
    label = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
    for i in range(0, len(color_list)):  # len(colors)
        color = color_list[i]
        index = np.where(np.all(img == color, axis=-1))
        label[index[0], index[1]] = i

    return label


def parsing_label2color(label):
    # convert the [face-parsing.Pytorch]-format label image to [face-parsing.Pytorch]-format RGB image
    color_list = [[0, 0, 0], [255, 0, 0], [150, 30, 150], [255, 65, 255],
                  [150, 80, 0], [170, 120, 65], [220, 180, 210], [255, 125, 125],
                  [200, 100, 100], [215, 175, 125], [125, 125, 125], [255, 150, 0],
                  [255, 255, 0], [0, 255, 255], [255, 225, 120], [125, 125, 255],
                  [0, 255, 0], [0, 0, 255], [0, 150, 80]
                  ]
    res = np.zeros((label.shape[0], label.shape[1], 3), dtype=np.uint8)
    for idx, color in enumerate(color_list):
        res[label == idx] = color

    return res


def celeba_label2parsinglabel(label):
    # convert the [CelebAMask-HQ]-format label image to [face-parsing.Pytorch]-format label image
    map_list = [0, 1, 6, 7, 4, 5, 3, 8, 9, 15, 2, 10, 11, 12, 17, 16, 18, 13, 14]
    res = label.copy()
    for i in range(0, len(map_list)):
        index = np.where(label == map_list[i])
        res[index[0], index[1]] = i

    return res


def normalize_SEAN(img):
    scale = 1.1
    img = cv2.resize(img, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
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


def get_face_from_parsing(im, parsing):
    # only face
    im = im[:, :, ::-1]
    # parsing = parsing[:, :, ::-1]    # rgb
    img = np.zeros((512, 512, 3), dtype=np.uint8)


    colors = [[255, 0, 0], [150, 30, 150], [255, 65, 255], [150, 80, 0], [170, 120, 65],
              [125, 125, 125], [255, 255, 0], [0, 255, 255], [255, 150, 0],
              [255, 125, 125], [200, 100, 100], [0, 0, 255]]
    for i in range(0, len(colors)):  
        color = colors[i]
        index = np.where(np.all(parsing == color, axis=-1))
        img[index[0], index[1], :] = im[index[0], index[1], :]


    img = img[:, :, ::-1]

    mask = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mask[mask > 0] = 255

    return img, mask


def get_face_from_parsing2(im, parsing):
    # add neck
    im = im[:, :, ::-1]
    parsing = parsing[:, :, ::-1]    # rgb
    img = np.zeros((512, 512, 3), dtype=np.uint8)


    colors = [[255, 0, 0], [150, 30, 150], [255, 65, 255], [150, 80, 0], [170, 120, 65],
              [125, 125, 125], [255, 255, 0], [0, 255, 255], [255, 150, 0],
              [255, 125, 125], [200, 100, 100], [255, 225, 120]]
    for i in range(0, len(colors)):  # len(colors)
        color = colors[i]
        index = np.where(np.all(parsing == color, axis=-1))
        img[index[0], index[1], :] = im[index[0], index[1], :]


    img = img[:, :, ::-1]

    mask = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mask[mask > 0] = 255

    return img, mask


def get_face_from_celeba_label(im, label):
    # with neck
    face_list = [1, 2, 4, 5, 6, 7, 10, 11, 12, 17, 8, 9]   # with ears and neck
    # face_list = [1, 2, 4, 5, 6, 7, 10, 11, 12, 17]    # with neck, no ears
    mask = np.zeros((512, 512, 3), dtype=np.uint8)
    for i in face_list:
        index = np.where(label == i)
        mask[index[0], index[1], :] = 1
    res_im = im * mask
    res_mask = mask[:, :, 0] * 255
    return res_im, res_mask


def get_face_from_celeba_label2(im, label):
    face_list = [1, 2, 4, 5, 6, 7, 10, 11, 12]   # no neck and ears
    mask = np.zeros((512, 512, 3), dtype=np.uint8)
    for i in face_list:
        index = np.where(label == i)
        mask[index[0], index[1], :] = 1
    res_im = im * mask
    res_mask = mask[:, :, 0] * 255
    return res_im, res_mask



def possion(src, dst, parsing):
    h, w, _ = dst.shape
    gray = cv2.cvtColor(parsing, cv2.COLOR_BGR2GRAY)
    index = np.where(gray > 0)
    parsing[index[0], index[1], :] = 255

    center_row = int((np.min(index[0]) + np.max(index[0]))/2)
    center_col = int((np.min(index[1]) + np.max(index[1]))/2)
    center = (center_col, center_row)

    # center = (h//2, w//2)

    # parsing = 255 - parsing
    img = cv2.seamlessClone(src, dst, parsing, center, cv2.NORMAL_CLONE)   # cv2.NORMAL_CLONE

    return img







