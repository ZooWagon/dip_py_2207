import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt


def Sobel(img_path):
    img = cv2.imread(img_path, 0)  # 读取图片
    # 边缘检测----Prewitt算子
    # 边缘检测----Sobel算子
    # cv2.Sobel(src, ddepth, dx, dy, dst, ksize, scale, delta,borderType)
    # dx和dy表示的是求导的阶数，0表示这个方向上没有求导

    x = cv2.Sobel(img, cv2.CV_16S, 1, 0)
    y = cv2.Sobel(img, cv2.CV_16S, 0, 1)

    # 转uint8
    absX = cv2.convertScaleAbs(x)
    absY = cv2.convertScaleAbs(y)

    # 加权
    Sobel = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)

    cv2.imwrite('./output/Sobel.jpg', Sobel)


'''
输入：一张灰度图 
输出：Sobel边缘检测的结果
'''
if __name__ == '__main__':
   Sobel(sys.argv[1])
