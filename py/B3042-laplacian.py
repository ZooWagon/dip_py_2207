import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt


def laplacian(img_path, ksize):
    img = cv2.imread(img_path, 0)  # 读取图片
    dst = cv2.Laplacian(img, cv2.CV_16S, ksize)

    # 数据格式转换
    Laplacian = cv2.convertScaleAbs(dst)

    cv2.imwrite('./output/Laplacian.jpg', Laplacian)

'''
输入：一张灰度图 ,滤波器大小ksize
输出：Laplacian边缘检测的结果
'''
if __name__ == '__main__':
    laplacian(sys.argv[1], eval(sys.argv[2]))
