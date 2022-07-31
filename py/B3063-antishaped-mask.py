import sys
import cv2
import numpy as np

import py2og


def Anti_sharpening_masking(img_path, sid):
    img = cv2.imread(img_path, 0)  # 读取图片
    m, n = img.shape[0], img.shape[1]

    # 利用局部平均法得到人为模糊后的的图像res
    kernel = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]], dtype=int)
    res = cv2.filter2D(img, cv2.CV_16S, kernel) / 9
    res = cv2.convertScaleAbs(res)

    ans = np.zeros((m, n), dtype=np.int32)

    for i in range(m):
        for j in range(n):
            ans[i, j] = int(img[i, j]) + 9 * (int(img[i, j]) - int(res[i, j]))
            if ans[i, j] > 255:
                ans[i, j] = 255
            elif ans[i][j] < 0:
                ans[i, j] = 0

    cv2.imwrite(py2og.outputFilePath+sid+'_out.jpg', ans)


'''
B3063
argv[1]: img name
argv[2]: sid
输入：img_path:图片地址
输出：反锐化掩膜法的结果
'''
if __name__ == '__main__':
    Anti_sharpening_masking(py2og.inputFilePath+sys.argv[1],sys.argv[2])
    py2og.updateDBwithFinishSignal(sys.argv[2])
