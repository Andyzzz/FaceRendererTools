
import numpy as np
import cv2
import glob
from PIL import Image
import dlib


# dlib检测关键点
def get_landmark_from_img(img):
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("./shape_predictor_68_face_landmarks.dat")
    POINTS_NUM_LANDMARK = 68

    dets = detector(img, 1)
    if len(dets) == 0:
        print("no face in the fake image")
        return None, -1
    ret = 0
    rectangle = dets[0]

    landmark_shape = predictor(img, rectangle)
    landmark_arr = np.zeros((68, 2))
    for i in range(0, POINTS_NUM_LANDMARK):
        landmark_arr[i, 0] = landmark_shape.part(i).x  # x
        landmark_arr[i, 1] = landmark_shape.part(i).y  # y

    return landmark_arr, ret

# 由dlib检测的landmark连成contour，绘制mask
def draw_landmark_contour(img_ori, landmark):

    img = img_ori.copy()

    landmark2 = np.zeros((27, 2))
    for i in range(0, 17):
        landmark2[i, 0] = landmark[i, 0]
        landmark2[i, 1] = landmark[i, 1]

    ind = 17
    for i in range(26, 16, -1):
        landmark2[ind, 0] = landmark[i, 0]
        landmark2[ind, 1] = landmark[i, 1]
        ind += 1


    # landmark2 = np.expand_dims(landmark2, axis=1)
    landmark2 = landmark2.reshape((-1, 1, 2)).astype(np.int32)
    print(landmark2.shape)

    contours = [landmark2]
    ind = 0
    mask = np.zeros_like(img)
    cv2.drawContours(mask, contours, ind, (255, 255, 255), -1)
    # 腐蚀
    kernal = np.ones((3, 3), np.uint8)
    mask = cv2.erode(mask, kernal, iterations=1)

    return mask


def normalize_SEAN(img):
    # resize images by scale, and move faces to get similar images of SEAN datasets or CelebA.
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


def combineImg(im1, im2, p1):
    # blend 2 images by weights
    res = cv2.addWeighted(im1, p1, im2, 1-p1, 0)
    return res


def get_stylegan_parsing_pair(image, color):
    # 修正因landmark稀疏导致的crop的mask与原有人脸轮廓不贴合的情况
    # 使crop的结果根据完整语义分割图中的人脸轮廓对齐

    landmark, ret = get_landmark_from_img(image)
    if ret < 0:
        return None, None, ret

    mask = draw_landmark_contour(image, landmark)

    mask[mask > 0] = 1.0
    crop = color * np.uint8(mask)

    # 对齐crop前后的脸部轮廓
    crop_label = parsing_label2celeba(parsing_Color2label(crop))
    full_label = parsing_label2celeba(parsing_Color2label(color))

    index1 = np.where(crop_label == 2)   # 鼻子
    if (not index1) or len(index1[0]) == 0:
        return crop, color, 0
    row = np.min(index1[0])  # 以前是鼻子的最上面一行，这里改成鼻子最上面一行再减去20

    index1 = np.where((crop_label == 6) | (crop_label == 7))  # 眉毛
    if (not index1) or len(index1[0]) == 0:
        row = row
    else:
        row = min(row, np.mean(index1[0]))  # row是眉毛的最下沿，如果没有眉毛，则是鼻子的最上沿

    # 以前没考虑到鼻子超出轮廓的情况，这里是新增的逻辑
    lis = [1, 2, 3, 4, 5, 6, 7, 11, 12]
    for ind in lis:
        index_face_full = np.where(full_label == ind)  # skin
        if (not index_face_full) or len(index_face_full[0]) == 0:
            continue
        index = ([i for i in index_face_full[0] if i > row],
                 [index_face_full[1][j] for j in range(0, len(index_face_full[1])) if index_face_full[0][j] > row])

        crop_label[index[0], index[1]] = full_label[index[0], index[1]]

    crop_final = parsing_label2color(celeba_label2parsinglabel(crop_label))

    return crop_final, color, 0



