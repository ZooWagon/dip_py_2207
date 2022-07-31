import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt


def Sobel(img_path, low, high):
    img = cv2.imread(img_path, 0)  # 读取图片
    gradx = cv2.Sobel(img, cv2.CV_16SC1, 1, 0)
    grady = cv2.Sobel(img, cv2.CV_16SC1, 0, 1)

    # 使用Canny函数处理图像，x,y分别是3求出来的梯度，低阈值50，高阈值150
    edge_output = cv2.Canny(gradx, grady, low, high)
    cv2.imwrite('./output/Canny.jpg', edge_output)


'''
输入：一张灰度图 ，低阈值，高阈值
输出：Canny边缘检测的结果
'''
if __name__ == '__main__':
    Sobel(sys.argv[1], eval(sys.argv[2]), eval(sys.argv[3]))
