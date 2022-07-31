import sys
import cv2
import py2og


def bgr2gray(img_path,sid):
    img = cv2.imread(img_path, 1)  # 读取图片
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(py2og.outputFilePath+sid+'_out.jpg', gray_image)


'''
B3012
argv[1]: img_name
argv[2]: sid
输入：彩色图片的地址
输出：灰度图片
'''
if __name__ == '__main__':
    bgr2gray(py2og.inputFilePath+sys.argv[1],sys.argv[2])
    py2og.updateDBwithFinishSignal(sys.argv[2])