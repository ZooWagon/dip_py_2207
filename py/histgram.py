import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt


def HistGraph(img_path):
    img = cv2.imread(img_path, 1)
    color = ["r", "g", "b"]  # 每个通道线的颜色
    # 因为用到cv2函数读取，但是用matplotlib库函数处理，所以应该转BGR格式为RGB格式
    img = img[:, :, ::-1]
    for index, c in enumerate(color):
        hist = cv2.calcHist([img], [index], None, [256], [0, 255])
        plt.plot(hist, color=c)
        plt.xlim([0, 255])
    plt.savefig("./output/HistGraph.jpg")


'''
输入：一张图片
输出：rgb三通道的灰度直方图
'''
if __name__ == '__main__':
    HistGraph(sys.argv[1])