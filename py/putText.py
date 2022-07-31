import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt


def putText(img_path, text, org, fontFace, fontScale, color, thickness=None, lineType=None):
    img = cv2.imread(img_path, 1)  # 读取图片
    cv2.putText(img, text, org, fontFace, fontScale, color, thickness, lineType)
    cv2.imwrite('./output/putText.jpg', img)


'''
输入:img_path, text, org, fontFace, fontScale, color, thickness, lineType
输出：在传入的img上添加一行文字
'''
if __name__ == '__main__':
    putText(sys.argv[1], sys.argv[2], eval(sys.argv[3]), eval(sys.argv[4]), eval(sys.argv[5]), eval(sys.argv[6]), eval(sys.argv[7]), eval(sys.argv[8]))
