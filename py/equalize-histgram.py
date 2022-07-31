import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt


def equalizeHist(img_path):
    img = cv2.imread(img_path, 1)
    r = cv2.equalizeHist(img[:, :, 0])
    g = cv2.equalizeHist(img[:, :, 1])
    b = cv2.equalizeHist(img[:, :, 2])
    res = cv2.merge([r, g, b])
    cv2.imwrite('/zzw/ogTest/output/equalizeHist.jpg', res)


'''
输入：一张图片
输出：直方图均衡化的结果
'''
if __name__ == '__main__':
    equalizeHist(sys.argv[1])