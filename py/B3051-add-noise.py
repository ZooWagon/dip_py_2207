import sys
import random
import cv2
import numpy as np

import py2og


def addNoisy(img_path, type, para1, para2, sid):
    img = cv2.imread(img_path)  # 读取图片
    out = np.zeros(img.shape, np.uint8)
    if type == 1:
        # 添加高斯噪声-------------------------------------------------------
        # 将图片的像素值归一化，存入矩阵中
        img = np.array(img / 255, dtype=float)
        # 生成正态分布的噪声，其中0表示均值，0.1表示方差
        noise = np.random.normal(para1, para2, img.shape)  # 参数：正态分布的均值和方差，可自定义
        # 将噪声叠加到图片上
        out = img + noise
        # 将图像的归一化像素值控制在0和1之间，防止噪声越界
        out = np.clip(out, 0.0, 1.0)
        # 将图像的像素值恢复到0到255之间
        out = np.uint8(out * 255)
        cv2.imwrite('./output/Gaussian_noisy.jpg', out)
    elif type == 2:
        # 添加椒盐噪声--------------------------------------------------------
        # 遍历图像，获取叠加噪声后的图像
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        for i in range(img_gray.shape[0]):
            for j in range(img_gray.shape[1]):
                rdn = random.random() % 255
                if rdn < para1:
                    # 添加椒噪声
                    out[i][j] = [0,0,0]
                elif rdn > para2:
                    # 添加盐噪声
                    out[i][j] = [255,255,255]
                else:
                    # 不添加噪声
                    out[i][j] = img[i][j]
    # cv2.imwrite('./output/saltnoisy.jpg', out)
    cv2.imwrite(py2og.outputFilePath+sid+'_out.jpg', out)


'''
B3051
argv[1]: img name
argv[2]: type
argv[3]: mu / threshold pepper
argv[4]: delte / threshold salt
argv[5]: sid
输入：img_path:图片地址
     type：噪声类型(1为高斯噪声，2为椒盐噪声）
     para1,para2：如果添加的是高斯噪声，para1为正态分布均值，para2为正态分布方差
                  如果添加的是椒盐噪声，para1为添加椒噪声阈值，para2为添加盐噪声阈值
输出：添加了噪声的图片
'''
if __name__ == '__main__':
    addNoisy(py2og.inputFilePath+sys.argv[1], eval(sys.argv[2]), eval(sys.argv[3]), eval(sys.argv[4]),sys.argv[5])
    py2og.updateDBwithFinishSignal(sys.argv[5])
