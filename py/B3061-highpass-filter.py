import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt

import py2og

def Highpass_filtering(img_path, type, D0, sid):
    print(D0)
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    img = cv2.imread(img_path)  # 读取图片
    b = img[:, :, 0]
    g = img[:, :, 1]
    r = img[:, :, 2]
    pltmod = cv2.merge([r, g, b])  # plt的顺序与cv2不同，显示正确结果需要重新组织顺序
    # 图片的中心（x,y)
    m, n = b.shape[0], b.shape[1]
    x = np.floor(m / 2)
    y = np.floor(n / 2)

    h = np.zeros((m, n))  # 理想高通滤波器
    if type == 1:
        for i in range(m):
            for j in range(n):
                D = np.sqrt((i - x) ** 2 + (j - y) ** 2)
                if D >= D0:
                    h[i, j] = 1
                else:
                    h[i, j] = 0

    elif type == 2:
        n0 = 2  # 参数
        for i in range(m):
            for j in range(n):
                D = np.sqrt((i - x) ** 2 + (j - y) ** 2)
                if D == 0:
                    h[i, j] = 0
                else:
                    h[i, j] = 1 / (1 + 0.414 * ((D0 / D) ** (2 * n0)))


    elif type == 3:
        n0 = 2  # 参数
        for i in range(m):
            for j in range(n):
                D = np.sqrt((i - x) ** 2 + (j - y) ** 2)
                if D == 0:
                    h[i, j] = 0
                else:
                    h[i, j] = np.exp(-0.347 * ((D0 / D) ** n0))

    fb = np.fft.fft2(b)
    fbshift = np.fft.fftshift(fb)
    fg = np.fft.fft2(g)
    fgshift = np.fft.fftshift(fg)
    fr = np.fft.fft2(r)
    frshift = np.fft.fftshift(fr)

    fbshift = fbshift * h
    fgshift = fgshift * h
    frshift = frshift * h

    ibshift = np.fft.ifftshift(fbshift)
    ibresult = np.fft.ifft2(ibshift)
    ibresult = np.uint8(np.abs(ibresult))
    igshift = np.fft.ifftshift(fgshift)
    igresult = np.fft.ifft2(igshift)
    igresult = np.uint8(np.abs(igresult))
    irshift = np.fft.ifftshift(frshift)
    irresult = np.fft.ifft2(irshift)
    irresult = np.uint8(np.abs(irresult))

    result1 = cv2.merge([irresult, igresult, ibresult])

    if type == 1:
        plt.imshow(result1)
        # plt.title('理想高通滤波')
        plt.axis('off')
        # plt.savefig('./output/理想高通滤波.jpg')
        plt.savefig(py2og.outputFilePath+sid+'_out.jpg')
    elif type == 2:
        plt.imshow(result1)
        # plt.title('巴特沃斯高通滤波')
        plt.axis('off')
        # plt.savefig('./output/巴特沃斯高通滤波.jpg')
        plt.savefig(py2og.outputFilePath+sid+'_out.jpg')
    elif type == 3:
        plt.imshow(result1)
        # plt.title('指数高通滤波')
        plt.axis('off')
        # plt.savefig('./output/指数高通滤波.jpg')
        plt.savefig(py2og.outputFilePath+sid+'_out.jpg')


'''
B3061
argv[1]: img name
argv[2]: type
argv[3]: shrethold [0,255]
argv[4]: sid
输入：img_path:图片地址
     type：滤波器类型（1:理想高通滤波、2:巴特沃斯高通滤波、3：指数高通滤波）
     D0：阈值
输出：高通滤波的结果
'''
if __name__ == '__main__':
    Highpass_filtering(py2og.inputFilePath+sys.argv[1], eval(sys.argv[2]), eval(sys.argv[3]),sys.argv[4])
    py2og.updateDBwithFinishSignal(sys.argv[4])