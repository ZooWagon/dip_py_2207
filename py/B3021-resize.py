import sys
import cv2
import py2og


def resize(img_path, shape, sid):
    img = cv2.imread(img_path, 1)
    img = cv2.resize(img, shape)
    cv2.imwrite(py2og.outputFilePath+sid+'_out.jpg', img)


'''
B3021
argv[1]: img name
argv[2]: target size like (x,y)
argv[3]: sid
输入：一张图片，图片大小
输出：对应大小的图片
'''
if __name__ == '__main__':
    resize(py2og.inputFilePath+sys.argv[1], eval(sys.argv[2]),sys.argv[3])
    py2og.updateDBwithFinishSignal(sys.argv[3])
