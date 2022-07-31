
import sys
import cv2
import numpy as np

import py2og


def translate(img_path, x, y, sid):
    img = cv2.imread(img_path, 1)
    M = np.float32([[1, 0, x], [0, 1, y]])
    result = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))
    cv2.imwrite(py2og.outputFilePath+sid+'_out.jpg', result)


'''
B3022
argv[1]: img name
argv[2]: move at x
argv[3]: move at y
argv[4]: sid
输入：一张图片，x方向移动数值，y方向移动数值 
输出：平移后的图片
'''
if __name__ == '__main__':
    translate(py2og.inputFilePath+sys.argv[1], eval(sys.argv[2]), eval(sys.argv[3]), sys.argv[4])
    py2og.updateDBwithFinishSignal(sys.argv[4])