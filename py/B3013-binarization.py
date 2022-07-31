import sys
import cv2
import py2og


def binarization(img_path, T=150, sid=""):
    img = cv2.imread(img_path, 0)  # 读取图片
    for i in range(0, img.shape[0]):
        for j in range(0, img.shape[1]):
            if img[i][j] >= T:  # 阈值，可自定义
                img[i][j] = 255
            else:
                img[i][j] = 0
    cv2.imwrite(py2og.outputFilePath+sid+'_out.jpg', img)


'''
B3013
argv[1]: img_name
argv[2]: threshold
argv[3]: sid
输入:img_path,阈值
输出：二值化图片
'''
if __name__ == '__main__':
    binarization(py2og.inputFilePath+sys.argv[1], eval(sys.argv[2]),sys.argv[3])
    py2og.updateDBwithFinishSignal(sys.argv[3])