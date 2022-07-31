import sys
import cv2
import numpy as np

import py2og


def Roberts(img_path, sid):
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
    # cv2.imwrite('./output/Roberts.jpg', Roberts)
    cv2.imwrite(py2og.outputFilePath+sid+'_out.jpg', Roberts)


def laplacian(img_path, ksize, sid):
    img = cv2.imread(img_path, 0)  # 读取图片
    dst = cv2.Laplacian(img, cv2.CV_16S, ksize)

    # 数据格式转换
    Laplacian = cv2.convertScaleAbs(dst)

    # cv2.imwrite('./output/Laplacian.jpg', Laplacian)
    cv2.imwrite(py2og.outputFilePath+sid+'_out.jpg', Laplacian)


def Prewitt(img_path, sid):
    img = cv2.imread(img_path, 0)  # 读取图片
    # 边缘检测----Prewitt算子
    kernelx = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]], dtype=int)
    kernely = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=int)

    x = cv2.filter2D(img, cv2.CV_16S, kernelx)
    y = cv2.filter2D(img, cv2.CV_16S, kernely)

    # 转uint8
    absX = cv2.convertScaleAbs(x)
    absY = cv2.convertScaleAbs(y)

    # 加权
    Prewitt = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)

    # cv2.imwrite('./output/Prewitt.jpg', Prewitt)
    cv2.imwrite(py2og.outputFilePath+sid+'_out.jpg', Prewitt)


def Sobel(img_path,sid):
    img = cv2.imread(img_path, 0)  # 读取图片
    # 边缘检测----Prewitt算子
    # 边缘检测----Sobel算子
    # cv2.Sobel(src, ddepth, dx, dy, dst, ksize, scale, delta,borderType)
    # dx和dy表示的是求导的阶数，0表示这个方向上没有求导

    x = cv2.Sobel(img, cv2.CV_16S, 1, 0)
    y = cv2.Sobel(img, cv2.CV_16S, 0, 1)

    # 转uint8
    absX = cv2.convertScaleAbs(x)
    absY = cv2.convertScaleAbs(y)

    # 加权
    Sobel = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)

    # cv2.imwrite('./output/Sobel.jpg', Sobel)
    cv2.imwrite(py2og.outputFilePath+sid+'_out.jpg', Sobel)


def Canny(img_path, low, high, sid):
    img = cv2.imread(img_path, 0)  # 读取图片
    gradx = cv2.Sobel(img, cv2.CV_16SC1, 1, 0)
    grady = cv2.Sobel(img, cv2.CV_16SC1, 0, 1)

    # 使用Canny函数处理图像，x,y分别是3求出来的梯度，低阈值50，高阈值150
    edge_output = cv2.Canny(gradx, grady, low, high)
    # cv2.imwrite('./output/Canny.jpg', edge_output)
    cv2.imwrite(py2og.outputFilePath+sid+'_out.jpg', edge_output)


def edge_detect(img_path, type, sid):
    if type == 1:
        Roberts(img_path, sid)
    elif type == 2:
        laplacian(img_path, 3, sid)
    elif type == 3:
        Prewitt(img_path, sid)
    elif type == 4:
        Sobel(img_path, sid)
    elif type == 5:
        Canny(img_path, 0, 255, sid)


'''
B3040
argv[1]: img name
argv[2]: type
argv[3]: sid
输入：一张灰度图 ,类型type
输出：边缘检测的结果
'''
if __name__ == '__main__':
    edge_detect(py2og.inputFilePath+sys.argv[1], eval(sys.argv[2]), sys.argv[3])
    py2og.updateDBwithFinishSignal(sys.argv[3])
