import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt


def ellipse(img_path, centerCoordinates, axesLength, angle, startAngle, endAngle, color, thickness):
    img = cv2.imread(img_path, 1)  # 读取图片
    cv2.ellipse(img, centerCoordinates, axesLength, angle, startAngle, endAngle, color, thickness)
    cv2.imwrite('./output/ellipse.jpg', img)


'''
输入:img_path, centerCoordinates, axesLength, angle, startAngle, endAngle, color, thickness
输出：在传入的img上添加一个椭圆
'''
if __name__ == '__main__':
    ellipse(sys.argv[1], eval(sys.argv[2]), eval(sys.argv[3]), eval(sys.argv[4]), eval(sys.argv[5]), eval(sys.argv[6]), eval(sys.argv[7]), eval(sys.argv[8]))
