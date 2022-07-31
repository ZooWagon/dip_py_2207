import sys
import cv2
import py2og


def mirror_flip(img_path, type, sid):
    img = cv2.imread(img_path, 1)
    result = cv2.flip(img, type)
    cv2.imwrite(py2og.outputFilePath+sid+'_out.jpg', result)


'''
B3023
argv[1]: img name
argv[2]: mirror type
argv[3]: sid
输入：图片地址，镜像类型(1,0,-1)
输出：type=1 水平镜像
     type=0 垂直镜像
     type=-1 对角镜像
'''
if __name__ == '__main__':
    mirror_flip(py2og.inputFilePath+sys.argv[1], eval(sys.argv[2]),sys.argv[3])
    py2og.updateDBwithFinishSignal(sys.argv[3])