import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt


def circle(img_path, center, radius, color, thickness):
    img = cv2.imread(img_path, 1)  # 读取图片
    cv2.circle(img, center, radius, color, thickness)
    cv2.imwrite('./output/circle.jpg', img)


'''
输入:img_path, center, radius, color, thickness
输出：在传入的img上添加一个圆形
'''
if __name__ == '__main__':
    circle(sys.argv[1], eval(sys.argv[2]), eval(sys.argv[3]), eval(sys.argv[4]), eval(sys.argv[5]))
