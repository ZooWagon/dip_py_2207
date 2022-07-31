import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt


def exp_grayscale_transformation(img, a, b, c):
    ans = np.zeros((img.shape[0], img.shape[1]))
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            ans[i][j] = np.power(b, c * (img[i][j] / 255 - a)) - 1
    return ans


def exp_grayscale_transformation_threeChannel(img_path, a, b, c):
    img = cv2.imread(img_path, 1)
    img = np.array(img, dtype=np.float64)
    # 三个通道分别变换
    r = exp_grayscale_transformation(img[:, :, 0], a, b, c)
    g = exp_grayscale_transformation(img[:, :, 1], a, b, c)
    b = exp_grayscale_transformation(img[:, :, 2], a, b, c)
    # 三通道合并
    res = cv2.merge([r, g, b])
    res = res*255
    cv2.imwrite('./output/exp_grayscale_transformation.jpg', res)


'''
输入：一张图片，参数a,b,c
输出：对数灰度变换的结果
'''
if __name__ == '__main__':
    exp_grayscale_transformation_threeChannel(sys.argv[1], eval(sys.argv[2]), eval(sys.argv[3]), eval(sys.argv[4]))