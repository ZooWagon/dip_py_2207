import sys
import cv2
import numpy as np

import py2og


def Differentiation(img_path, T, sid):
    img = cv2.imread(img_path,0)  # 读取图片
    m, n = img.shape[0], img.shape[1]

    ans = np.zeros((m, n), dtype=np.uint8)

    kernelx = np.array([[-1, 0], [0, 1]], dtype=int)
    kernely = np.array([[0, -1], [1, 0]], dtype=int)

    x = cv2.filter2D(img, cv2.CV_16S, kernelx)
    y = cv2.filter2D(img, cv2.CV_16S, kernely)

    absX = cv2.convertScaleAbs(x)
    absY = cv2.convertScaleAbs(y)

    Roberts = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)

    # 背景保留
    for i in range(m):
        for j in range(n):
            if Roberts[i, j] >= T:
                ans[i, j] = Roberts[i, j]
            else:
                ans[i, j] = img[i, j]

    cv2.imwrite(py2og.outputFilePath+sid+'_out.jpg', ans)


'''
B3062
argv[1]: img name
argv[2]: shrethold [0,255]
argv[3]: sid
输入：img_path:图片地址
     T：阈值
输出：微分法的结果
'''
if __name__ == '__main__':
    Differentiation(py2og.inputFilePath+sys.argv[1], eval(sys.argv[2]),sys.argv[3])
    py2og.updateDBwithFinishSignal(sys.argv[3])
