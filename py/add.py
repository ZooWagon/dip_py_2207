import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt


def add(img1_path, img2_path):
    img1 = cv2.imread(img1_path, 1)
    img2 = cv2.imread(img2_path, 1)
    result = cv2.add(img1, img2)
    cv2.imwrite('./output/add.jpg', result)


'''
输入：两张图片（大小相同） 
输出：求和之后的结果
'''
if __name__ == '__main__':
    add(sys.argv[1], sys.argv[2])
