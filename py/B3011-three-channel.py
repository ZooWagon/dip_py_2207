import sys
import cv2
import py2og


def three_channel(img_path,sid):
    img = cv2.imread(img_path, 1)  # 读取图片
    b = img[:, :, 0]
    g = img[:, :, 1]
    r = img[:, :, 2]
    cv2.imwrite(py2og.outputFilePath+sid+'_b.jpg', b)
    cv2.imwrite(py2og.outputFilePath+sid+'_g.jpg', g)
    cv2.imwrite(py2og.outputFilePath+sid+'_r.jpg', r)


'''
B3011
argv[1]: img_name
argv[2]: sid
输出：b,g,r三通道共三张图片
'''
if __name__ == '__main__':
    three_channel(py2og.inputFilePath+sys.argv[1],sys.argv[2])
    py2og.updateDBwithFinishSignal(sys.argv[2])
