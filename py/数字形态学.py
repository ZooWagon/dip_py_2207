import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt

'''
形态学操作
cv2.morphologyEx(src, op, kernel)  
src：输入图像，即源图像；
op: 表示形态学运算的类型，有以下几个可选项
MORPH_ERODE：“腐蚀”；
MORPH_DILATE：“膨胀”；
MORPH_OPEN：开运算；
MORPH_CLOSE：闭运算；
MORPH_TOPHAT：“顶帽”  f(top) = f - f(open) ；
MORPH_BLACKHAT：“黑帽”“底帽” f(black) = f(close) - f ；
MORPH_GRADIENT：形态学梯度 f(grad) = f(dilate) - f(erode) ；
kernel：形态学运算的内核。若为NULL时，表示的是默认使用参考点位于中心3 x 3的核。

'''


def Digital_morphology(img_path, type, shape, size):
    kernel = cv2.getStructuringElement(shape, size, (-1, -1))  # 5x5交叉结构元
    img = cv2.imread(img_path)  # 读取图片
    res = cv2.morphologyEx(img, type, kernel)
    cv2.imwrite('./output/Digital_morphology.jpg', res)


'''
输入：img_path:图片地址
     type：形态学操作类型，见上方注释，直接传入CV2.xxx的字符串
     shape:结构元形状，cv2.MORPH_RECT;  cv2.MORPH_CROSS;  cv2.MORPH_ELLIPSE;  分别对应矩形结构元、交叉形结构元和椭圆形结构元。
     size:结构元大小，例如(5,5)
输出：数字形态学操作的结果
'''
if __name__ == '__main__':
    Digital_morphology(sys.argv[1], eval(sys.argv[2]), eval(sys.argv[3]), eval(sys.argv[4]))

