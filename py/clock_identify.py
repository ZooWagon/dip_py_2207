import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
import jaydebeapi
import py2og


def clock_identify(img_path, sid):
    clock = cv2.imread(img_path, 1)  # 读取图片

    # 0、高斯滤波去噪
    clock = cv2.GaussianBlur(clock, (5, 5), 0, 0)

    # 1、灰度化
    GrayImage = cv2.cvtColor(clock, cv2.COLOR_BGR2GRAY)

    # 2、Roberts边缘提取
    kernelx = np.array([[-1, 0], [0, 1]], dtype=int)
    kernely = np.array([[0, -1], [1, 0]], dtype=int)
    x = cv2.filter2D(GrayImage, cv2.CV_16S, kernelx)
    y = cv2.filter2D(GrayImage, cv2.CV_16S, kernely)
    absX = cv2.convertScaleAbs(x)
    absY = cv2.convertScaleAbs(y)
    Roberts = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)

    # 3、二值化
    for i in range(0, Roberts.shape[0]):
        for j in range(0, Roberts.shape[1]):
            if Roberts[i][j] >= 30:
                Roberts[i][j] = 255
            else:
                Roberts[i][j] = 0

    # 4、霍夫直线检测
    linesP = cv2.HoughLinesP(image=Roberts, rho=1, theta=np.pi / 180, threshold=150, minLineLength=clock.shape[0] / 6,
                             maxLineGap=10)  # (在canny边缘检测的结果上进行边缘链接)

    for i_P in linesP:
        for x1, y1, x2, y2 in i_P:
            cv2.line(clock, (x1, y1), (x2, y2), (0, 0, 255), 3)

    # 5、霍夫圆检测
    circlesP = cv2.HoughCircles(image=Roberts, method=cv2.HOUGH_GRADIENT, dp=2, minDist=500, param1=100, param2=100,
                                minRadius=200, maxRadius=500)
    circlesP = circlesP.reshape(-1, 3)
    circlesP = np.uint16(np.around(circlesP))

    print(circlesP)
    for i in circlesP:
        cv2.circle(clock, (i[0], i[1]), i[2], (0, 0, 255), 5)  # 画圆
        cv2.imwrite('./output/'+sid+'_clock.jpg', clock)
        cv2.imwrite('./output/'+sid+'_edge.jpg', Roberts)

    def getDist(point1, point2):
        return np.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)

    def angle(v1, v2):
        cosvalue = (v1[0] * v2[0] + v1[1] * v2[1]) / (
                    np.sqrt(v1[0] ** 2 + v1[1] ** 2) * np.sqrt(v2[0] ** 2 + v2[1] ** 2))
        return np.arccos(cosvalue)

    def isSimilar(line1, line2, center):
        nearpoint1 = [line1[0], line1[1]]
        farpoint1 = [line1[2], line1[3]]
        nearpoint2 = [line2[0], line2[1]]
        farpoint2 = [line2[2], line2[3]]
        if getDist(nearpoint1, center) > getDist(farpoint1, center):
            nearpoint1, farpoint1 = farpoint1, nearpoint1
        if getDist(nearpoint2, center) > getDist(farpoint2, center):
            nearpoint2, farpoint2 = farpoint2, nearpoint2
        vector1 = [farpoint1[0] - nearpoint1[0], farpoint1[1] - nearpoint1[1]]
        vector2 = [farpoint2[0] - nearpoint2[0], farpoint2[1] - nearpoint2[1]]
        if angle(vector1, vector2) < 0.09:
            return True
        else:
            return False

    line_group = [[], [], []]
    center = [circlesP[0][0], circlesP[0][1]]
    print(center)
    for i_P in linesP:
        for group in line_group:
            if len(group) == 0:
                group.append(i_P[0])
                break
            else:
                if (isSimilar(i_P[0], group[0], center)):
                    group.append(i_P[0])
                    break

    for group in line_group:
        for line in group:
            print(line)
        print("--------------")

    def getAvgLength(group):
        ans = 0
        for line in group:
            ans = ans + getDist([line[0], line[1]], [line[2], line[3]])
        return ans / len(group)

    def getAngle(line):
        nearpoint = [line[0], line[1]]
        farpoint = [line[2], line[3]]
        if getDist(nearpoint, center) > getDist(farpoint, center):
            nearpoint, farpoint = farpoint, nearpoint
        vector = [farpoint[0] - nearpoint[0], farpoint[1] - nearpoint[1]]
        if vector[0] == 0 and vector[1] > 0:
            return np.pi
        if vector[0] > 0 and vector[1] > 0:
            return np.pi / 2 + np.arctan(vector[1] / vector[0])
        if vector[0] > 0 and vector[1] == 0:
            return np.pi / 2
        if vector[0] > 0 and vector[1] < 0:
            return np.arctan(vector[0] / (-vector[1]))
        if vector[0] == 0 and vector[1] < 0:
            return 0
        if vector[0] < 0 and vector[1] < 0:
            return 2 * np.pi - np.arctan((-vector[0]) / (-vector[1]))
        if vector[0] < 0 and vector[1] == 0:
            return 1.5 * np.pi
        if vector[0] < 0 and vector[1] > 0:
            return np.pi + np.arctan((-vector[0]) / vector[1])

    class pointer:
        def __init__(self, group):
            self.length = getAvgLength(group)
            self.angle = getAngle(group[0])

    clockhands = []
    for group in line_group:
        clockhands.append(pointer(group))

    clockhands.sort(key=lambda x: x.length)

    hour = (clockhands[0].angle / (2 * np.pi)) * 12
    minites = (clockhands[1].angle / (2 * np.pi)) * 60
    seconds = (clockhands[2].angle / (2 * np.pi)) * 60
    seconds = int(seconds)
    minites = int(minites)
    hour = int(hour)

    iden_time="{:02d}".format(hour) + ":" + "{:02d}".format(minites) + ":" + "{:02d}".format(seconds) 
    return iden_time


'''
argv[1]: sid
argv[2]: input image name
'''
if __name__ == '__main__':
    iden_time=clock_identify(py2og.inputFilePath+sys.argv[2],sys.argv[1])
    # write file
    # with open(py2og.outputFilePath+sys.argv[1]+"_time.txt", 'w') as f:
    #     f.write(iden_time+'\n')
    #     f.close()
    # update db
    conn = jaydebeapi.connect(py2og.dirver, py2og.url, [py2og.user, py2og.password], py2og.jarFile)
    curs = conn.cursor()
    sqlStr = "update submission set status='已完成', para8='"+ iden_time + "' where sid='"+sys.argv[1]+"' ;"
    curs.execute(sqlStr)
    # result = curs.fetchall()
    # print(result)
    curs.close()
    conn.close()