import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt


def Roberts(img_path):
    img = cv2.imread(img_path, 0)  # 读取图片
    kernelx = np.array([[-1, 0], [0, 1]], dtype=int)
    kernely = np.array([[0, -1], [1, 0]], dtype=int)

    # filter后会有负值，还有会大于255的值。而原图像是uint8，会有截断。因此要使用16位有符号的数据类型，即cv2.CV_16S。
    x = cv2.filter2D(img, cv2.CV_16S, kernelx)
    y = cv2.filter2D(img, cv2.CV_16S, kernely)

    # 用convertScaleAbs()函数将其转回原来的uint8形式。否则将无法显示图像，而只是一副灰色的窗口
    absX = cv2.convertScaleAbs(x)
    absY = cv2.convertScaleAbs(y)

    # 用cv2.addWeighted(...)函数将其组合起来，其中，alpha是第一幅图片中元素的权重，beta是第二个的权重，gamma是加到最后结果上的一个值。
    Roberts = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)
    cv2.imwrite('./output/Roberts.jpg', Roberts)


'''
输入：一张灰度图  
输出：Roberts边缘检测的结果
'''
if __name__ == '__main__':
    Roberts(sys.argv[1])
