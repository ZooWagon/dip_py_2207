import sys
import cv2
import numpy as np

import py2og


def warpAffine(img_path, matSrc, matDst, sid):
    img = cv2.imread(img_path, 1)
    height, width = img.shape[:2]
    M = cv2.getAffineTransform(np.float32(matSrc), np.float32(matDst))  # 生成矩阵
    result = cv2.warpAffine(img, M, (width, height))
    cv2.imwrite(py2og.outputFilePath+sid+'_out.jpg', result)


'''
B3026
argv[1]: img name
argv[2]: matSrc
argv[3]: matDst
argv[4]: sid
输入：一张图片，旋转前三个点的坐标matSrc，旋转后三个点的坐标matDst
输出：仿射变换后的图片
'''
if __name__ == '__main__':
    warpAffine(py2og.inputFilePath+sys.argv[1], eval(sys.argv[2]), eval(sys.argv[3]),sys.argv[4])
    py2og.updateDBwithFinishSignal(sys.argv[4])