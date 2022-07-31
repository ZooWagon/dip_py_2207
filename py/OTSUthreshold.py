import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt


def OTSUthreshold(img_path, thresh, maxval):
    img = cv2.imread(img_path, 0)
    ret, threshans = cv2.threshold(img, thresh, maxval, type=cv2.THRESH_OTSU)
    plt.imshow(threshans, cmap='gray')
    plt.title('OTSUthreshold')
    plt.axis('off')
    plt.savefig("./output/OTSUthreshold.jpg")


'''
输入：一张灰度图片(传进彩色图片会被转为灰度图），门限值,结果图片像素的最大值
输出：OTSU灰度级门限化的结果
'''
if __name__ == '__main__':
    OTSUthreshold(sys.argv[1], eval(sys.argv[2]), eval(sys.argv[3]))