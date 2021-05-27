import cv2  # Import the OpenCV library
import numpy as np  # Import Numpy library
# import matplotlib.pyplot as plt  # Import matplotlib functionality
import sys  # Enables the passing of arguments
import glob
from get_pair_parsing2 import *
import os.path as osp


#================================= hist matching ================================
def calculate_cdf(histogram):
    """
    This method calculates the cumulative distribution function
    :param array histogram: The values of the histogram
    :return: normalized_cdf: The normalized cumulative distribution function
    :rtype: array
    """
    # Get the cumulative sum of the elements
    cdf = histogram.cumsum()

    # Normalize the cdf
    normalized_cdf = cdf / float(cdf.max())

    return normalized_cdf


def calculate_lookup(src_cdf, ref_cdf):
    """
    This method creates the lookup table
    :param array src_cdf: The cdf for the source image
    :param array ref_cdf: The cdf for the reference image
    :return: lookup_table: The lookup table
    :rtype: array
    """
    lookup_table = np.zeros(256)
    lookup_val = 0
    for src_pixel_val in range(len(src_cdf)):
        # lookup_val
        for ref_pixel_val in range(len(ref_cdf)):
            if ref_cdf[ref_pixel_val] >= src_cdf[src_pixel_val]:
                lookup_val = ref_pixel_val
                break
        lookup_table[src_pixel_val] = lookup_val
    return lookup_table


def match_histograms(src_image, ref_image, mask_src, mask_ref):
    """
    This method matches the source image histogram to the
    reference signal
    :param image src_image: The original source image
    :param image  ref_image: The reference image
    :return: image_after_matching
    :rtype: image (array)
    """
    # Split the images into the different color channels
    # b means blue, g means green and r means red
    src_b, src_g, src_r = cv2.split(src_image)
    ref_b, ref_g, ref_r = cv2.split(ref_image)

    src_hist_blue = cv2.calcHist([src_image], [0], mask_src, [256], [0, 256])  
    src_hist_green = cv2.calcHist([src_image], [1], mask_src, [256], [0, 256])
    src_hist_red = cv2.calcHist([src_image], [2], mask_src, [256], [0, 256])
    ref_hist_blue = cv2.calcHist([ref_image], [0], mask_ref, [230], [0, 230])
    ref_hist_green = cv2.calcHist([ref_image], [1], mask_ref, [230], [0, 230])
    ref_hist_red = cv2.calcHist([ref_image], [2], mask_ref, [230], [0, 230])


    # Compute the normalized cdf for the source and reference image
    src_cdf_blue = calculate_cdf(src_hist_blue)
    src_cdf_green = calculate_cdf(src_hist_green)
    src_cdf_red = calculate_cdf(src_hist_red)
    ref_cdf_blue = calculate_cdf(ref_hist_blue)
    ref_cdf_green = calculate_cdf(ref_hist_green)
    ref_cdf_red = calculate_cdf(ref_hist_red)

    # Make a separate lookup table for each color
    blue_lookup_table = calculate_lookup(src_cdf_blue, ref_cdf_blue)
    green_lookup_table = calculate_lookup(src_cdf_green, ref_cdf_green)
    red_lookup_table = calculate_lookup(src_cdf_red, ref_cdf_red)

    # Use the lookup function to transform the colors of the original
    # source image
    blue_after_transform = cv2.LUT(src_b, blue_lookup_table)
    green_after_transform = cv2.LUT(src_g, green_lookup_table)
    red_after_transform = cv2.LUT(src_r, red_lookup_table)

    # blue_after_transform[mask_src == 0] = src_b[mask_src == 0]
    # green_after_transform[mask_src == 0] = src_g[mask_src == 0]
    # red_after_transform[mask_src == 0] = src_r[mask_src == 0]

    # kernel = np.ones((11, 11), np.uint8)
    # mask_src = cv2.erode(mask_src, kernel)

    mask_blur = cv2.blur(mask_src, (15, 15))/255  
    # mask_blur = mask_src/255

    blue_after_transform = blue_after_transform * mask_blur + src_b * (1 - mask_blur)
    green_after_transform = green_after_transform * mask_blur + src_g * (1 - mask_blur)
    red_after_transform = red_after_transform * mask_blur + src_r * (1 - mask_blur)

    # Put the image back together
    image_after_matching = cv2.merge([
        blue_after_transform, green_after_transform, red_after_transform])
    image_after_matching = cv2.convertScaleAbs(image_after_matching)

    return image_after_matching


def mask_image(image, mask):
    """
    This method overlays a mask on top of an image
    :param image image: The color image that you want to mask
    :param image mask: The mask
    :return: masked_image
    :rtype: image (array)
    """

    # Split the colors into the different color channels
    blue_color, green_color, red_color = cv2.split(image)

    # Resize the mask to be the same size as the source image
    resized_mask = cv2.resize(
        mask, (image.shape[1], image.shape[0]), cv2.INTER_NEAREST)

    # Normalize the mask
    normalized_resized_mask = resized_mask / float(255)

    # Scale the color values
    blue_color = blue_color * normalized_resized_mask
    blue_color = blue_color.astype(int)
    green_color = green_color * normalized_resized_mask
    green_color = green_color.astype(int)
    red_color = red_color * normalized_resized_mask
    red_color = red_color.astype(int)

    # Put the image back together again
    merged_image = cv2.merge([blue_color, green_color, red_color])
    masked_image = cv2.convertScaleAbs(merged_image)
    return masked_image


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


    # cv2.imshow("im", img)
    # cv2.waitKey()
    return img


def hist_match(image_src, mask_src, image_ref, mask_ref):
    # Calculate the matched image
    output_image = match_histograms(image_src, image_ref, mask_src, mask_ref)

    return output_image


#================================== face swap =========================================

def faceswap(im1, im2, parsing):
    gray = cv2.cvtColor(parsing, cv2.COLOR_BGR2GRAY)
    index = np.where(gray > 0)
    im2[index[0], index[1], :] = im1[index[0], index[1], :]
    return im2


def faceswap2(im1, im2, parsing):
    # 在下面的应用里parsing是celeba的label
    if len(parsing.shape) == 3:
        gray = cv2.cvtColor(parsing, cv2.COLOR_BGR2GRAY)
        index = np.where(gray > 0)
    else:
        index = np.where(parsing > 0)

    mask = np.zeros((512, 512, 3))
    mask[index[0], index[1], :] = 1

    kernel = np.ones((11, 11), np.uint8)   # 11, 11
    mask = cv2.erode(mask, kernel)  # 先扩张mask
    mask_blur = cv2.blur(mask, (11, 11))  # 再模糊边缘  # 15, 15

    # index_nose = np.where(parsing == 2)
    # if index_nose and len(index_nose[0]) > 0:
    #     row = np.min(index_nose[0])
    #     mask_new = np.zeros((512, 512, 3))
    #     mask_new[0: row, :, :] = mask_blur[0: row, :, :]
    #     mask_new[row: 512, :, :] = mask[row: 512, :, :]
    #     mask_blur = mask_new

    img = im1 * mask_blur + im2 * (1 - mask_blur)
    img = img.astype(np.uint8)

    return img, mask_blur*255


def faceswap3(im1, im2, parsing1, parsing2):
    # 0331重新写的，跟原来的不同之处是把额头的部分也加入换脸，旧的版本是只换额头以下的部分
    # parsing1是3DMM人脸的mask， parsing2是SEAN人脸的mask
    # 在下面的应用里parsing是彩色的parsing
    face1, mask1 = get_face_from_parsing(im1, parsing1)
    face2, mask2 = get_face_from_parsing(im2, parsing2)

    mask2 = 255 - mask2
    index1 = np.where(mask1 > 0)
    mask = np.zeros((512, 512, 3))
    mask[index1[0], index1[1], :] = 1
    index2 = np.where(mask2 > 0)
    mask[index2[0], index2[1], :] = 0

    kernel = np.ones((11, 11), np.uint8)  # 11, 11
    mask = cv2.erode(mask, kernel)  # 先扩张mask　　# 注意这里一些细小的黑色区域经过腐蚀以后会变大
    # cv2.imshow("img", mask)
    # cv2.waitKey()
    mask_blur = cv2.blur(mask, (11, 11))  # 再模糊边缘  # 15, 15

    # # ================ 对侧脸情况下加上这一段，对mask的不同区域的模糊程度不同 ============================
    # # # 先简单修改成上半部分为mask_blur，下半部分为mask_src
    # kernel3 = np.ones((21, 21), np.uint8)  # 11, 11
    # mask3 = cv2.erode(mask, kernel3)
    # mask_blur2 = cv2.blur(mask3, (19, 19))
    #
    # kernel4 = np.ones((9, 9), np.uint8)
    # mask4 = cv2.erode(mask, kernel4)
    # mask_blur3 = cv2.blur(mask4, (21, 21))
    #
    # mask_new = np.zeros_like(mask)
    # mask_new[0:200, :, :] = mask_blur3[0:200, :, :]
    # mask_new[200:512, :, :] = mask_blur[200:512, :, :]
    # mask_new[:, 0:200, :] = mask_blur3[:, 0:200, :]
    # mask_blur = mask_new
    # # ============================================================================================

    img = im1 * mask_blur + im2 * (1 - mask_blur)
    img = img.astype(np.uint8)

    # return img, mask1
    return img, mask_blur * 255


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


def generate_facescape_front3():
    lis = sorted(glob.glob("/home/zhang/zydDataset/faceRendererData/TUFrontFaceImages_light2/*_*"))
    test_img_dir = "/home/zhang/PycharmProjects/SEAN/datasets/facescape_front2/test_img/"
    tgt_img_dir = "/home/zhang/PycharmProjects/SEAN/datasets/facescape_front3/test_img/"

    expList = ['1_neutral', '2_smile', '3_mouth_stretch', '4_anger', '5_jaw_left', '6_jaw_right', '7_jaw_forward',
               '8_mouth_left', '9_mouth_right',
               '10_dimpler', '11_chin_raiser', '12_lip_puckerer', '13_lip_funneler', '14_sadness', '15_lip_roll',
               '16_grin', '17_cheek_blowing',
               '18_eye_closed', '19_brow_raiser', '20_brow_lower']
    for i in range(0, len(lis)):
        for num in range(0, len(expList)):
            test_img_name = lis[i].split("/")[-1] + "_" + expList[num] + ".png"
            subdir = lis[i].split("/")[-1]
            for light_i in range(0, 9):
                light_img_name = expList[num] + "_" + str(light_i) + ".png"
                light_img = np.array(Image.open(osp.join(lis[i], light_img_name)))
                light_img = normalize_SEAN(light_img)
                index = np.where(np.all(light_img < 255, axis=-1))
                test_img = np.array(Image.open(osp.join(test_img_dir, test_img_name)))
                test_img[index[0], index[1], :] = light_img[index[0], index[1], :]

                Image.fromarray(test_img).save(tgt_img_dir + subdir + "_" + light_img_name)

                label = np.array(Image.open(osp.join(test_img_dir.replace("test_img", "test_label"),
                                                     test_img_name.replace("test_img", "test_label"))))
                Image.fromarray(label).save(tgt_img_dir.replace("test_img", "test_label") + subdir + "_" + light_img_name)


def generate_facescape_front4():
    lis = sorted(glob.glob("/home/zhang/zydDataset/faceRendererData/TUFrontFaceImages_light3/*_*"))
    test_img_dir = "/home/zhang/PycharmProjects/SEAN/datasets/facescape_front2/test_img/"
    tgt_img_dir = "/home/zhang/PycharmProjects/SEAN/datasets/facescape_front4/test_img/"

    expList = ['1_neutral', '2_smile', '3_mouth_stretch', '4_anger', '5_jaw_left', '6_jaw_right', '7_jaw_forward',
               '8_mouth_left', '9_mouth_right',
               '10_dimpler', '11_chin_raiser', '12_lip_puckerer', '13_lip_funneler', '14_sadness', '15_lip_roll',
               '16_grin', '17_cheek_blowing',
               '18_eye_closed', '19_brow_raiser', '20_brow_lower']
    for i in range(0, len(lis)):
        for num in range(0, len(expList)):
            test_img_name = lis[i].split("/")[-1] + "_" + expList[num] + ".png"    # front2中的均匀光照的图像
            subdir = lis[i].split("/")[-1]
            for light_i in range(0, 9):
                light_img_name = expList[num] + "_" + str(light_i) + ".png"  # light3中的强光照的图像

                # light3中的有的图像像素会达到255，所以这里的mask来源来自于light2的图像
                imgname = osp.join(lis[i].replace("light3", "light2"), light_img_name)
                if not osp.isfile(imgname):
                    continue
                img = np.array(Image.open(imgname))

                # 读取light3中的图像

                light_img = np.array(Image.open(osp.join(lis[i], light_img_name)))
                light_img = normalize_SEAN(light_img)


                img = normalize_SEAN(img)
                index = np.where(np.all(img < 255, axis=-1))

                test_img = np.array(Image.open(osp.join(test_img_dir, test_img_name)))
                test_img[index[0], index[1], :] = light_img[index[0], index[1], :]

                Image.fromarray(test_img).save(tgt_img_dir + subdir + "_" + light_img_name)

                label = np.array(Image.open(osp.join(test_img_dir.replace("test_img", "test_label"),
                                                     test_img_name.replace("test_img", "test_label"))))
                Image.fromarray(label).save(tgt_img_dir.replace("test_img", "test_label") + subdir + "_" + light_img_name)

                # 保存对应的style图片，要求1024x1024， jpg格式
                style_img_dir = tgt_img_dir.replace("test_img", "test_img_style")
                test_img2 = cv2.resize(test_img, (0, 0), fx=2, fy=2)
                Image.fromarray(test_img2).save(style_img_dir + subdir + "_" + light_img_name.replace("png", "jpg"))





def SEAN_swap():
    # ================================= 不带pose的路径 ==============================================
    # crop_parsing_dir = "/home/zhang/zydDataset/faceRendererData/testResults/1_face-parsing/0324/0324_addhair/"
    #
    # syn_img_dir = "/home/zhang/zydDataset/faceRendererData/testResults/3_SEAN/0324_addhair_3/CelebA-HQ_pretrained/test_latest/images/synthesized_image/"
    # syn_label_dir = "/home/zhang/PycharmProjects/SEAN/datasets/facescape_0324_addhair_3/test_label/"
    # facescape_img_dir = "/home/zhang/PycharmProjects/SEAN/datasets/facescape_0324_addhair_3/test_img/"
    #
    # swap_save_dir = "/home/zhang/zydDataset/faceRendererData/testResults/4_faceswap/0324_addhair_3/"
    crop_parsing_dir = "/run/user/1000/gvfs/smb-share:server=cite-3d.local,share=share/zhangyidi/FaceRendererData/testResults/1_face-parsing/0517/crop/"

    syn_img_dir = "/run/user/1000/gvfs/smb-share:server=cite-3d.local,share=share/zhangyidi/FaceRendererData/testResults/3_SEAN/results/0517/CelebA-HQ_pretrained/test_latest/images/synthesized_image/"
    syn_label_dir = "/run/user/1000/gvfs/smb-share:server=cite-3d.local,share=share/zhangyidi/FaceRendererData/testResults/3_SEAN/datasets/0517/test_label/"
    facescape_img_dir = "/run/user/1000/gvfs/smb-share:server=cite-3d.local,share=share/zhangyidi/FaceRendererData/testResults/3_SEAN/datasets/0517/test_img/"

    swap_save_dir = "/run/user/1000/gvfs/smb-share:server=cite-3d.local,share=share/zhangyidi/FaceRendererData/testResults/4_faceswap/0517/"
    # ==============================================================================================


    # # =================== 带有pose的路径 ===========================================================
    # pose_ind = "2/"
    # crop_parsing_dir = "/home/zhang/zydDataset/faceRendererData/testResults/1_face-parsing/0325/crop/" + pose_ind
    #
    # syn_img_dir = "/home/zhang/zydDataset/faceRendererData/testResults/3_SEAN/0325/" + pose_ind + "CelebA-HQ_pretrained/test_latest/images/synthesized_image/"
    # syn_label_dir = "/home/zhang/PycharmProjects/SEAN/datasets/facescape_0325/" + pose_ind + "test_label/"
    # facescape_img_dir = "/home/zhang/PycharmProjects/SEAN/datasets/facescape_0325/" + pose_ind + "test_img/"
    #
    # swap_save_dir = "/home/zhang/zydDataset/faceRendererData/testResults/4_faceswap/0325/" + pose_ind
    # # =============================================================================================

    if not osp.exists(swap_save_dir):
        os.makedirs(swap_save_dir)
    lis = sorted(glob.glob(crop_parsing_dir + "*.png"))
    for i in range(0, len(lis)):  # len(lis)
        # name = lis[i].split("/")[-2] + "_" + lis[i].split("/")[-1]
        name = lis[i].split("/")[-1]
        print(name)
        # facescape
        ref_color = cv2.imread(lis[i])[:, :, ::-1]
        # ref_color = normalize_SEAN(ref_color)
        ref_label0 = parsing_Color2label(ref_color)
        # ref_label0 = normalize_SEAN(ref_label0)
        ref_label = parsing_label2celeba(ref_label0)
        # ref_color2 = celeba_label2color(ref_label)


        ref_im = cv2.imread(facescape_img_dir + name)    # bgr
        ref_face, ref_mask = get_face_from_celeba_label(ref_im[:, :, ::-1], ref_label)

        src_label = cv2.imread(syn_label_dir + name, 0)
        src_im = cv2.imread(syn_img_dir + name)
        src_im = cv2.resize(src_im, (0, 0), fx=2, fy=2)   # bgr

        src_face, src_mask = get_face_from_celeba_label(src_im[:, :, ::-1], src_label)

        out = hist_match(src_im, src_mask, ref_im, ref_mask)  # hist_match的输入都是bgr
        # ==================== 过滤亮点的部分 =====================
        # hsv = cv2.cvtColor(out, cv2.COLOR_BGR2HSV)
        # mask_face = np.zeros((512, 512), dtype=np.uint8)
        # mask_face[src_label == 1] = 1
        # mask_light = np.zeros((512, 512), dtype=np.uint8)
        #
        # mean_h = np.mean(hsv[:, :, 0]*mask_face)
        # mean_v = np.mean(hsv[:, :, 2] * mask_face)
        # mask_light[hsv[:, :, 2] * mask_face > mean_v + 150] = 1   # 182
        # mask_light[hsv[:, :, 0]*mask_face > mean_h + 9] = 1
        # # mask_light[hsv[:, :, 0]*mask_face <= 10] = 0
        # # cv2.imwrite("/home/zhang/zydDataset/faceRendererData/mask_light/" + name, mask_light*255)
        # out2 = cv2.medianBlur(out, 25)
        #
        # index = np.where(mask_light > 0)
        # out[index[0], index[1], :] = out2[index[0], index[1], :]
        # ========================================================

        swap, mask = faceswap2(ref_im, out, ref_label)

        # # 保存中间过程的out和mask，作为毕设插图
        # outdir = "/home/zhang/zydDataset/faceRendererData/testResults/4_faceswap/0324_addhair3_histmatch/"
        # maskdir = "/home/zhang/zydDataset/faceRendererData/testResults/4_faceswap/0324_addhair3_mask/"
        # cv2.imwrite(outdir + name, out)
        # cv2.imwrite(maskdir + name, mask)

        cv2.imwrite(swap_save_dir + name, swap)

def SEAN_swap2():
    # test 直接加权融合，不进行直方图匹配
    # ================================= 不带pose的路径 ==============================================
    crop_parsing_dir = "/home/zhang/zydDataset/faceRendererData/testResults/1_face-parsing/0324/crop/"

    syn_img_dir = "/home/zhang/zydDataset/faceRendererData/testResults/3_SEAN/0324/CelebA-HQ_pretrained/test_latest/images/synthesized_image/"
    syn_label_dir = "/home/zhang/PycharmProjects/SEAN/datasets/facescape_0324/test_label/"
    facescape_img_dir = "/home/zhang/PycharmProjects/SEAN/datasets/facescape_0324/test_img/"

    swap_save_dir = "/home/zhang/zydDataset/faceRendererData/testResults/4_faceswap/0324_3/"
    # ==============================================================================================
    if not osp.exists(swap_save_dir):
        os.makedirs(swap_save_dir)
    lis = sorted(glob.glob(crop_parsing_dir + "*.png"))
    for i in range(0, len(lis)):  # len(lis)
        # name = lis[i].split("/")[-2] + "_" + lis[i].split("/")[-1]
        name = lis[i].split("/")[-1]
        print(name)
        # facescape
        ref_color = cv2.imread(lis[i])[:, :, ::-1]    # crop的彩色语义分割图
        # ref_color = normalize_SEAN(ref_color)
        ref_label0 = parsing_Color2label(ref_color)
        # ref_label0 = normalize_SEAN(ref_label0)
        ref_label = parsing_label2celeba(ref_label0)
        # ref_color2 = celeba_label2color(ref_label)


        ref_im = cv2.imread(facescape_img_dir + name)    # bgr
        ref_face, ref_mask = get_face_from_celeba_label(ref_im[:, :, ::-1], ref_label)

        src_label = cv2.imread(syn_label_dir + name, 0)
        src_im = cv2.imread(syn_img_dir + name)
        src_im = cv2.resize(src_im, (0, 0), fx=2, fy=2)   # bgr

        src_face, src_mask = get_face_from_celeba_label(src_im[:, :, ::-1], src_label)

        # out = hist_match(src_im, src_mask, ref_im, ref_mask)  # hist_match的输入都是bgr
        # swap, mask = faceswap2(ref_im, out, ref_label)

        mask_blur = cv2.blur(ref_mask, (15, 15)) / 255  # 这里需要修改
        # # 先简单修改成上半部分为mask_blur，下半部分为mask_src
        # mask_new = np.zeros_like(mask_blur)
        # mask_new[0:256, :] = mask_blur[0:256, :]
        # mask_new[256:512, :] = ref_mask[256:512, :]/255
        # mask_blur = mask_new

        swap = np.zeros_like(ref_im)
        swap[:, :, 0] = ref_im[:, :, 0] * mask_blur + src_im[:, :, 0] * (1 - mask_blur)
        swap[:, :, 1] = ref_im[:, :, 1] * mask_blur + src_im[:, :, 1] * (1 - mask_blur)
        swap[:, :, 2] = ref_im[:, :, 2] * mask_blur + src_im[:, :, 2] * (1 - mask_blur)


        cv2.imwrite(swap_save_dir + name, swap)


def SEAN_swap3():
    # test 仅使用泊松融合
    # ================================= 不带pose的路径 ==============================================
    crop_parsing_dir = "/home/zhang/zydDataset/faceRendererData/testResults/1_face-parsing/0324/crop/"

    syn_img_dir = "/home/zhang/zydDataset/faceRendererData/testResults/3_SEAN/0324/CelebA-HQ_pretrained/test_latest/images/synthesized_image/"
    syn_label_dir = "/home/zhang/PycharmProjects/SEAN/datasets/facescape_0324/test_label/"
    facescape_img_dir = "/home/zhang/PycharmProjects/SEAN/datasets/facescape_0324/test_img/"

    swap_save_dir = "/home/zhang/zydDataset/faceRendererData/testResults/4_faceswap/0324_4/"
    # ==============================================================================================
    if not osp.exists(swap_save_dir):
        os.makedirs(swap_save_dir)
    lis = sorted(glob.glob(crop_parsing_dir + "*.png"))
    for i in range(0, len(lis)):  # len(lis)
        # name = lis[i].split("/")[-2] + "_" + lis[i].split("/")[-1]
        name = lis[i].split("/")[-1]
        print(name)
        # facescape
        ref_color = cv2.imread(lis[i])[:, :, ::-1]    # crop的彩色语义分割图
        # ref_color = normalize_SEAN(ref_color)

        ref_im = cv2.imread(facescape_img_dir + name)    # bgr

        src_im = cv2.imread(syn_img_dir + name)
        src_im = cv2.resize(src_im, (0, 0), fx=2, fy=2)   # bgr

        swap = possion(ref_im, src_im, ref_color)
        cv2.imwrite(swap_save_dir + name, swap)


def SEAN_swap4():
    # test 直接加权融合，不进行直方图匹配
    # ================================= 不带pose的路径 ==============================================
    crop_parsing_dir = "/home/zhang/zydDataset/faceRendererData/testResults/1_face-parsing/0324/crop/"

    syn_img_dir = "/home/zhang/zydDataset/faceRendererData/testResults/3_SEAN/0324/CelebA-HQ_pretrained/test_latest/images/synthesized_image/"
    syn_label_dir = "/home/zhang/PycharmProjects/SEAN/datasets/facescape_0324/test_label/"
    facescape_img_dir = "/home/zhang/PycharmProjects/SEAN/datasets/facescape_0324/test_img/"

    swap_save_dir = "/home/zhang/zydDataset/faceRendererData/testResults/4_faceswap/0324_3/"
    # ==============================================================================================
    if not osp.exists(swap_save_dir):
        os.makedirs(swap_save_dir)
    lis = sorted(glob.glob(crop_parsing_dir + "*.png"))
    for i in range(0, len(lis)):  # len(lis)
        # name = lis[i].split("/")[-2] + "_" + lis[i].split("/")[-1]
        name = lis[i].split("/")[-1]
        print(name)
        # facescape
        ref_color = cv2.imread(lis[i])[:, :, ::-1]    # crop的彩色语义分割图
        # ref_color = normalize_SEAN(ref_color)
        ref_label0 = parsing_Color2label(ref_color)
        # ref_label0 = normalize_SEAN(ref_label0)
        ref_label = parsing_label2celeba(ref_label0)
        # ref_color2 = celeba_label2color(ref_label)


        ref_im = cv2.imread(facescape_img_dir + name)    # bgr
        ref_face, ref_mask = get_face_from_celeba_label(ref_im[:, :, ::-1], ref_label)

        src_label = cv2.imread(syn_label_dir + name, 0)
        src_im = cv2.imread(syn_img_dir + name)
        src_im = cv2.resize(src_im, (0, 0), fx=2, fy=2)   # bgr

        src_face, src_mask = get_face_from_celeba_label(src_im[:, :, ::-1], src_label)

        out = hist_match(src_im, src_mask, ref_im, ref_mask)  # hist_match的输入都是bgr
        swap, mask = faceswap2(ref_im, out, ref_label)

        cv2.imwrite(swap_save_dir + name, swap)


def SEAN_swap5():
    # 尝试新的想法：在换脸的时候把原图的额头部分加上
    # ================================= 不带pose的路径 ==============================================
    crop_parsing_dir = "/home/zhang/zydDataset/faceRendererData/testResults/1_face-parsing/0324/0324_addhair/"
    full_parsing_dir = "/home/zhang/zydDataset/faceRendererData/testResults/1_face-parsing/0324/full/"
    syn_img_dir = "/home/zhang/zydDataset/faceRendererData/testResults/3_SEAN/0324_addhair_3/CelebA-HQ_pretrained/test_latest/images/synthesized_image/"
    syn_label_dir = "/home/zhang/PycharmProjects/SEAN/datasets/facescape_0324_addhair_3/test_label/"
    facescape_img_dir = "/home/zhang/PycharmProjects/SEAN/datasets/facescape_0324_addhair_3/test_img/"

    swap_save_dir = "/home/zhang/zydDataset/faceRendererData/testResults/4_faceswap/0324_addhair_4_poisson/"
    # ==============================================================================================

    # # =================== 带有pose的路径 ===========================================================
    # pose_ind = "5/"
    # crop_parsing_dir = "/home/zhang/zydDataset/faceRendererData/testResults/1_face-parsing/0325/crop/" + pose_ind
    # full_parsing_dir = "/home/zhang/zydDataset/faceRendererData/testResults/1_face-parsing/0325/full/" + pose_ind
    #
    # syn_img_dir = "/home/zhang/zydDataset/faceRendererData/testResults/3_SEAN/0325/" + pose_ind + "CelebA-HQ_pretrained/test_latest/images/synthesized_image/"
    # syn_label_dir = "/home/zhang/PycharmProjects/SEAN/datasets/facescape_0325/" + pose_ind + "test_label/"
    # facescape_img_dir = "/home/zhang/PycharmProjects/SEAN/datasets/facescape_0325/" + pose_ind + "test_img/"
    #
    # swap_save_dir = "/home/zhang/zydDataset/faceRendererData/testResults/4_faceswap/0326/" + pose_ind
    # # =============================================================================================

    if not osp.exists(swap_save_dir):
        os.makedirs(swap_save_dir)
    lis = sorted(glob.glob(crop_parsing_dir + "*.png"))
    for i in range(0, len(lis)):  # len(lis)
        # name = lis[i].split("/")[-2] + "_" + lis[i].split("/")[-1]
        name = lis[i].split("/")[-1]
        print(name)
        # facescape
        ref_color = cv2.imread(lis[i])[:, :, ::-1]   # crop的parsing
        # ref_color = normalize_SEAN(ref_color)
        ref_label0 = parsing_Color2label(ref_color)  # crop的label
        # ref_label0 = normalize_SEAN(ref_label0)
        ref_label = parsing_label2celeba(ref_label0)  # crop的label
        # ref_color2 = celeba_label2color(ref_label)
        full_ref_color = cv2.imread(lis[i].replace(crop_parsing_dir, full_parsing_dir))[:, :, ::-1]
        full_ref_label = parsing_label2celeba(parsing_Color2label(full_ref_color))   # full的label


        ref_im = cv2.imread(facescape_img_dir + name)    # bgr
        ref_face, ref_mask = get_face_from_celeba_label(ref_im[:, :, ::-1], ref_label)

        src_label = cv2.imread(syn_label_dir + name, 0)
        src_im = cv2.imread(syn_img_dir + name)
        src_im = cv2.resize(src_im, (0, 0), fx=2, fy=2)   # bgr

        src_face, src_mask = get_face_from_celeba_label(src_im[:, :, ::-1], src_label)

        out = hist_match(src_im, src_mask, ref_im, ref_mask)  # hist_match的输入都是bgr

        # swap, mask = faceswap3(ref_im, out, full_ref_color, parsing_label2color(celeba_label2parsinglabel(src_label)))

        # poisson
        swap = possion(ref_im, src_im, ref_color)

        # # 保存中间过程的out和mask，作为毕设插图
        # outdir = "/home/zhang/zydDataset/faceRendererData/testResults/4_faceswap/0324_addhair4_histmatch/"
        # maskdir = "/home/zhang/zydDataset/faceRendererData/testResults/4_faceswap/0324_addhair4_mask/"
        # cv2.imwrite(outdir + name, out)
        # cv2.imwrite(maskdir + name, mask)

        cv2.imwrite(swap_save_dir + name, swap)


def possion2(src, dst, mask):
    h, w, _ = dst.shape
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.erode(mask, kernel)
    index = np.where(mask > 0)
    parsing = np.zeros((512, 512, 3), dtype=np.uint8)
    parsing[index[0], index[1], :] = 255

    center_row = int((np.min(index[0]) + np.max(index[0])) / 2)
    center_col = int((np.min(index[1]) + np.max(index[1])) / 2)
    center = (center_col, center_row)

    # center = (h//2, w//2)

    img = cv2.seamlessClone(src, dst, parsing, center, cv2.NORMAL_CLONE)  # cv2.NORMAL_CLONE

    return img

def SEAN_swap6():
    # 尝试新的想法：在换脸的时候把原图的额头部分加上 只匹配额头
    # # ================================= 不带pose的路径 ==============================================
    # crop_parsing_dir = "/home/zhang/zydDataset/faceRendererData/testResults/1_face-parsing/0324/crop/"
    # full_parsing_dir = "/home/zhang/zydDataset/faceRendererData/testResults/1_face-parsing/0324/full/"
    # syn_img_dir = "/home/zhang/zydDataset/faceRendererData/testResults/3_SEAN/0324/CelebA-HQ_pretrained/test_latest/images/synthesized_image/"
    # syn_label_dir = "/home/zhang/PycharmProjects/SEAN/datasets/facescape_0324/test_label/"
    # facescape_img_dir = "/home/zhang/PycharmProjects/SEAN/datasets/facescape_0324/test_img/"
    #
    # swap_save_dir = "/home/zhang/zydDataset/faceRendererData/testResults/4_faceswap/0324_2/"
    # # ==============================================================================================

    # =================== 带有pose的路径 ===========================================================
    pose_ind = "2/"
    crop_parsing_dir = "/home/zhang/zydDataset/faceRendererData/testResults/1_face-parsing/0325/crop/" + pose_ind
    full_parsing_dir = "/home/zhang/zydDataset/faceRendererData/testResults/1_face-parsing/0325/full/" + pose_ind

    syn_img_dir = "/home/zhang/zydDataset/faceRendererData/testResults/3_SEAN/0325/" + pose_ind + "CelebA-HQ_pretrained/test_latest/images/synthesized_image/"
    syn_label_dir = "/home/zhang/PycharmProjects/SEAN/datasets/facescape_0325/" + pose_ind + "test_label/"
    facescape_img_dir = "/home/zhang/PycharmProjects/SEAN/datasets/facescape_0325/" + pose_ind + "test_img/"

    swap_save_dir = "/home/zhang/zydDataset/faceRendererData/testResults/4_faceswap/0326_2/" + pose_ind
    # =============================================================================================

    if not osp.exists(swap_save_dir):
        os.makedirs(swap_save_dir)
    lis = sorted(glob.glob(crop_parsing_dir + "*.png"))
    for i in range(0, len(lis)):  # len(lis)
        # name = lis[i].split("/")[-2] + "_" + lis[i].split("/")[-1]
        name = lis[i].split("/")[-1]
        print(name)
        # facescape
        ref_color = cv2.imread(lis[i])[:, :, ::-1]   # crop的parsing
        # ref_color = normalize_SEAN(ref_color)
        ref_label0 = parsing_Color2label(ref_color)  # crop的label
        # ref_label0 = normalize_SEAN(ref_label0)
        ref_label = parsing_label2celeba(ref_label0)  # crop的label
        # ref_color2 = celeba_label2color(ref_label)
        full_ref_color = cv2.imread(lis[i].replace(crop_parsing_dir, full_parsing_dir))[:, :, ::-1]
        full_ref_label = parsing_label2celeba(parsing_Color2label(full_ref_color))   # full的label


        ref_im = cv2.imread(facescape_img_dir + name)    # bgr
        ref_face, ref_mask = get_face_from_celeba_label(ref_im[:, :, ::-1], ref_label)     # cropface1
        ref_face_full, ref_mask_full = get_face_from_celeba_label(ref_im[:, :, ::-1], full_ref_label)    # fullface1

        src_label = cv2.imread(syn_label_dir + name, 0)
        src_im = cv2.imread(syn_img_dir + name)
        src_im = cv2.resize(src_im, (0, 0), fx=2, fy=2)   # bgr

        src_face, src_mask = get_face_from_celeba_label(src_im[:, :, ::-1], src_label)      # fullface2

        # 先匹配全脸
        out = hist_match(src_im, src_mask, ref_im, ref_mask)  # hist_match的输入都是bgr

        # 再单独匹配额头
        index1 = np.where(ref_mask > 0)
        ref_mask_new = ref_mask_full.copy()
        ref_mask_new[index1[0], index1[1]] = 0     # 额头１的区域   单通道

        src_face, src_mask = get_face_from_celeba_label2(src_im[:, :, ::-1], src_label)
        index1 = np.where(ref_mask_full > 0)
        src_mask_new = src_mask.copy()
        src_mask_new[index1[0], index1[1]] = 0     # 额头２的区域　　　单通道
        # out = hist_match(out, src_mask_new, ref_im, ref_mask_new)  # hist_match的输入都是bgr

        swap, mask = faceswap3(ref_im, out, full_ref_color, parsing_label2color(celeba_label2parsinglabel(src_label)))

        # 把额头２的区域拿出来扩大一圈，再泊松融合进换脸的图像
        swap = possion2(out, swap, src_mask_new)

        cv2.imwrite(swap_save_dir + name, swap)



def SEAN_swap_withlight():
    crop_parsing_dir = "/home/zhang/zydDataset/faceRendererData/TUFront_parsing_crop/"

    syn_img_dir = "/home/zhang/PycharmProjects/SEAN/results_front4_skin/CelebA-HQ_pretrained/test_latest/images/synthesized_image/"
    syn_label_dir = "/home/zhang/PycharmProjects/SEAN/datasets/facescape_front4/test_label/"
    facescape_img_dir = "/home/zhang/PycharmProjects/SEAN/datasets/facescape_front4/test_img/"

    swap_save_dir = "/home/zhang/zydDataset/faceRendererData/facescape_front4_swap/"
    lis = sorted(glob.glob(crop_parsing_dir + "*_*/*.png"))
    for i in range(0, 1600):  # len(lis)
        name = lis[i].split("/")[-2] + "_" + lis[i].split("/")[-1]
        print(name)
        for light_i in range(0, 9):
            # facescape
            name_light = lis[i].split("/")[-2] + "_" + lis[i].split("/")[-1].split(".")[0] + "_" + str(light_i) + ".png"
            if not os.path.isfile(osp.join(facescape_img_dir, name_light)):
                print("not exist")
                print(osp.join(facescape_img_dir, name_light))
                continue
            ref_color = cv2.imread(lis[i])[:, :, ::-1]
            # ref_color = normalize_SEAN(ref_color)
            ref_label0 = parsing_Color2label(ref_color)
            ref_label0 = normalize_SEAN(ref_label0)
            ref_label = parsing_label2celeba(ref_label0)
            # ref_color2 = celeba_label2color(ref_label)

            ref_im = cv2.imread(facescape_img_dir + name_light)    # bgr
            ref_face, ref_mask = get_face_from_celeba_label(ref_im[:, :, ::-1], ref_label)

            src_label = cv2.imread(syn_label_dir + name, 0)
            src_im = cv2.imread(syn_img_dir + name_light)
            src_im = cv2.resize(src_im, (0, 0), fx=2, fy=2)   # bgr

            src_face, src_mask = get_face_from_celeba_label(src_im[:, :, ::-1], src_label)

            out = hist_match(src_im, src_mask, ref_im, ref_mask)  # hist_match的输入都是bgr
            swap, mask = faceswap2(ref_im, out, ref_label)
            # swap, mask = faceswap2(ref_im, src_im, ref_label)

            if not os.path.exists(swap_save_dir):
                os.makedirs(swap_save_dir)
            cv2.imwrite(swap_save_dir + name_light, swap)



if __name__ == '__main__':
    # print(__doc__)
    # main()
    # cv2.destroyAllWindows()

    # swap rawscan with SEAN
    SEAN_swap()




