import sys
import cv2
import numpy as np
 
import py2og


def zoom(img_path, x, y, sid):
    img = cv2.imread(img_path, 1)
    M = np.float32([[x, 0, 0], [0, y, 0]])  # 图片宽变为原来的x倍，高变为原来的y倍
    result = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))
    cv2.imwrite(py2og.outputFilePath+sid+'_out.jpg', result)


'''
B3024
argv[1]: img name
argv[2]: width scale
argv[3]: height scale
argv[4]: sid
输入：一张图片，宽度缩放比例，高度缩放比例
输出：缩放后的图片
'''
if __name__ == '__main__':
    zoom(py2og.inputFilePath+sys.argv[1], eval(sys.argv[2]), eval(sys.argv[3]), sys.argv[4])
    py2og.updateDBwithFinishSignal(sys.argv[4])