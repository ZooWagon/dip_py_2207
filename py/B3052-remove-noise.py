import sys
import cv2
import numpy as np

import py2og


def Arithmetic_mean_filtering(img_path,):
    img = cv2.imread(img_path,0)  # 读取图片
    result1 = np.zeros(img.shape, np.uint8)
    # 算数均值滤波
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            sum = 0
            for m in range(-1, 2):
                for n in range(-1, 2):
                    if 0 <= i + m < img.shape[0] and 0 <= j + n < img.shape[1]:
                        # 像素值求和
                        sum += img[i + m][j + n]
            result1[i][j] = (sum / 9).astype(np.int32)

    #cv2.imwrite('./output/Arithmetic_mean_filtering.jpg', result1)
    return result1

def Maximum_mean_filtering(img_path):
    img = cv2.imread(img_path,0)  # 读取图片
    result = np.zeros(img.shape, np.uint8)
    # 最大值滤波器
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            # 最大值滤波器
            max_ = 0
            for m in range(-1, 2):
                for n in range(-1, 2):
                    if 0 <= i + m < img.shape[0] and 0 <= j + n < img.shape[1]:
                        # 通过比较判断是否需要更新最大值
                        if img[i + m][j + n] > max_:
                            # 更新最大值
                            max_ = img[i + m][j + n]
            result[i][j] = max_

    #cv2.imwrite('./output/Maximum_filter.jpg', result)
    return result


def GaussianBlur_filtering(img_path):
    img = cv2.imread(img_path)  # 读取图片
    blur = cv2.GaussianBlur(img, (5, 5), 0, 0)
    # cv2.imwrite('./output/GaussianBlur_filter.jpg', blur)
    return blur


def medianBlur_filtering(img_path):
    img = cv2.imread(img_path)  # 读取图片
    median = cv2.medianBlur(img, 5)
    #cv2.imwrite('./output/median_filter.jpg', median)
    return median


def Geometric_mean_filtering(img_path):
    img = cv2.imread(img_path,0)  # 读取图片
    result1 = np.zeros(img.shape, np.uint8)
    # 几何均值滤波
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            ji = 1.0
            for m in range(-1, 2):
                for n in range(-1, 2):
                    if 0 <= i + m < img.shape[0] and 0 <= j + n < img.shape[1]:
                        ji = ji * img[i + m][j + n]
            result1[i][j] = pow(ji, 1 / 9)
    # cv2.imwrite('./output/Geometric_mean_filtering.jpg', result1)
    return result1


def filtering(img_path, type, sid):
    if type == 1:
        out=Arithmetic_mean_filtering(img_path)
    elif type == 2:
        out=Geometric_mean_filtering(img_path)
    elif type == 3:
        out=Maximum_mean_filtering(img_path)
    elif type == 4:
        out=GaussianBlur_filtering(img_path)
    elif type == 5:
        out=medianBlur_filtering(img_path)
    cv2.imwrite(py2og.outputFilePath+sid+'_out.jpg', out)


'''
B3052
argv[1]: img name
argv[2]: type
argv[3]: sid
输入：img_path:添加了噪声的图片
     type:滤波种类
输出：滤波的结果
'''
if __name__ == '__main__':
    filtering(py2og.inputFilePath+sys.argv[1], eval(sys.argv[2]),sys.argv[3])
    py2og.updateDBwithFinishSignal(sys.argv[3])