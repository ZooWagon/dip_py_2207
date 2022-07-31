import sys
import cv2

import py2og


def rotate(img_path, center, angle, scale, sid):
    img = cv2.imread(img_path, 1)
    M = cv2.getRotationMatrix2D(center, angle, scale)  # 使用内置函数构建矩阵
    result = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))
    cv2.imwrite(py2og.outputFilePath+sid+'_out.jpg', result)


'''
B3025
argv[1]: img name
argv[2]: center
argv[3]: angle
argv[4]: scale
argv[5]: sid
输入：一张图片，旋转中心，旋转角
输出：旋转后的图片
'''
if __name__ == '__main__':
    rotate(py2og.inputFilePath+sys.argv[1], eval(sys.argv[2]), eval(sys.argv[3]), eval(sys.argv[4]), sys.argv[5])
    py2og.updateDBwithFinishSignal(sys.argv[5])
