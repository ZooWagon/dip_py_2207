import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt


def rectangle(img_path, pt1, pt2, color, thickness):
    img = cv2.imread(img_path, 1)  # 读取图片
    cv2.rectangle(img, pt1, pt2, color, thickness)
    cv2.imwrite('./output/rectangle.jpg', img)


'''
输入:img_path,pt1,pt2,color,thickness
输出：在传入的img上添加一个矩形
'''
if __name__ == '__main__':
    rectangle(sys.argv[1], eval(sys.argv[2]), eval(sys.argv[3]), eval(sys.argv[4]), eval(sys.argv[5]))
