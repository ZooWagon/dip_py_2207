import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt


def Prewitt(img_path):
    img = cv2.imread(img_path, 0)  # 读取图片
    # 边缘检测----Prewitt算子
    kernelx = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]], dtype=int)
    kernely = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=int)

    x = cv2.filter2D(img, cv2.CV_16S, kernelx)
    y = cv2.filter2D(img, cv2.CV_16S, kernely)

    # 转uint8
    absX = cv2.convertScaleAbs(x)
    absY = cv2.convertScaleAbs(y)

    # 加权
    Prewitt = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)


    cv2.imwrite('./output/Prewitt.jpg', Prewitt)
'''
输入：一张灰度图 
输出：Prewitt边缘检测的结果
'''
if __name__ == '__main__':
    Prewitt(sys.argv[1])