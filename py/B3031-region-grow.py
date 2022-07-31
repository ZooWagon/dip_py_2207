import sys
import cv2
import numpy as np

import py2og


class Point(object):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def getX(self):
        return self.x

    def getY(self):
        return self.y


def getGrayDiff(img, currentPoint, tmpPoint):  # 获得两点像素之差的绝对值
    return abs(int(img[currentPoint.x, currentPoint.y]) - int(img[tmpPoint.x, tmpPoint.y]))


def selectConnects(p):  # 8邻域或者4邻域
    if p != 0:
        connects = [Point(-1, -1), Point(0, -1), Point(1, -1), Point(1, 0), Point(1, 1), \
                    Point(0, 1), Point(-1, 1), Point(-1, 0)]
    else:
        connects = [Point(0, -1), Point(1, 0), Point(0, 1), Point(-1, 0)]
    return connects


def regionGrow(img_path, x, y, thresh, p, sid):
    img = cv2.imread(img_path, 0)  # 读取图片
    height, weight = img.shape
    seedMark = np.zeros(img.shape)
    seedList = []
    seedList.append(Point(x, y))
    label = 1
    connects = selectConnects(p)
    while (len(seedList) > 0):
        currentPoint = seedList.pop(0)

        seedMark[currentPoint.x, currentPoint.y] = label
        for i in range(len(connects)):
            tmpX = currentPoint.x + connects[i].x
            tmpY = currentPoint.y + connects[i].y
            if tmpX < 0 or tmpY < 0 or tmpX >= height or tmpY >= weight:
                continue
            grayDiff = getGrayDiff(img, currentPoint, Point(tmpX, tmpY))
            if grayDiff < thresh and seedMark[tmpX, tmpY] == 0:
                seedMark[tmpX, tmpY] = label
                seedList.append(Point(tmpX, tmpY))
    # cv2.imwrite('./output/regionGrow.jpg', seedMark*255)
    cv2.imwrite(py2og.outputFilePath+sid+'_out.jpg', seedMark*255)

'''
B3031
argv[1]: img name
argv[2]: seed x
argv[3]: seed y
argv[4]: threshold
argv[5]: domain type: 0-four, 1-eight
argv[6]: sid
输入：img_path:图片地址
     seeds:初始种子，是一个元素类型为Point的列表，如[ Point(0, -1),  Point(1, 0),Point(0, 1), Point(-1, 0)]
     thresh：门限值，两个坐标像素值小于thresh归为一类
     p：等于0算法基于四领域，等于1算法基于八领域
输出：二值图片，白色部分为初始种子所生长出来的区域，黑色为另一区域
'''
if __name__ == '__main__':
    regionGrow(py2og.inputFilePath+sys.argv[1], eval(sys.argv[2]), eval(sys.argv[3]), eval(sys.argv[4]), eval(sys.argv[5]),sys.argv[6])
    py2og.updateDBwithFinishSignal(sys.argv[6])
