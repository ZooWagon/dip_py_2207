import sys
import cv2
import numpy as np

import py2og


def Houghlines(img_path, threshold, minLineLength, maxLineGap, sid):
    img = cv2.imread(img_path, 1)  # 读取图片
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    linesP = cv2.HoughLinesP(gray_image, 1, np.pi / 180, threshold, None, minLineLength, maxLineGap)
    for i_P in linesP:
        for x1, y1, x2, y2 in i_P:
            cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 1)
    cv2.imwrite(py2og.outputFilePath+sid+'_out.jpg', img)

'''
B3046
argv[1]: img name
argv[2]: threshold 检测一条直线所需最少的曲线交点
argv[3]: minLineLength 组成一条直线的最少点的数量
argv[4]: maxLineGap 认为在一条直线上的亮点的最大距离
argv[5]: sid
输入：一张任意某个边缘提取的结果图片，threshold， minLineLength, maxLineGap
输出：边缘连接的结果的结果
'''
if __name__ == '__main__':
    Houghlines(py2og.inputFilePath+sys.argv[1], eval(sys.argv[2]), eval(sys.argv[3]), eval(sys.argv[4]),sys.argv[5])
    py2og.updateDBwithFinishSignal(sys.argv[5])