import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt


def polylines(img_path, data, isClosed, color, thickness):
    img = cv2.imread(img_path, 1)  # 读取图片
    pts = np.array(data, np.int32)
    pts = pts.reshape((-1, 1, 2))
    cv2.polylines(img, [pts], isClosed, color, thickness)
    cv2.imwrite('./output/polylines.jpg', img)


'''
输入:img_path, pts, isClosed, color, thickness
输出：在传入的img上添加一个多边形
'''
if __name__ == '__main__':
    polylines(sys.argv[1], eval(sys.argv[2]), eval(sys.argv[3]), eval(sys.argv[4]), eval(sys.argv[5]))
