import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt


def linear_grayscale_transformation(img, a, b, c, d):
    ans = np.zeros((img.shape[0], img.shape[1]))
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if img[i][j] > b:
                ans[i][j] = d
            elif a <= img[i][j] <= b:
                ans[i][j] = (d - c) / (b - a) * (img[i][j] - a) + c
            else:
                ans[i][j] = c
    return ans / 255


def linear_grayscale_transformation_threeChannel(img_path, a, b, c, d):
    img = cv2.imread(img_path, 1)
    img = np.array(img, dtype=np.float64)
    # 三个通道分别变换
    r = linear_grayscale_transformation(img[:, :, 0], a, b, c, d)
    g = linear_grayscale_transformation(img[:, :, 1], a, b, c, d)
    b = linear_grayscale_transformation(img[:, :, 2], a, b, c, d)
    # 三通道合并
    res = cv2.merge([r, g, b])
    res = res*255
    cv2.imwrite('./output/linear_grayscale_transformation.jpg', res)


'''
输入：一张图片，参数a,b,c,d
输出：线性灰度变换的结果
'''
if __name__ == '__main__':
    linear_grayscale_transformation_threeChannel(sys.argv[1], eval(sys.argv[2]), eval(sys.argv[3]), eval(sys.argv[4]),
                                                 eval(sys.argv[5]))
