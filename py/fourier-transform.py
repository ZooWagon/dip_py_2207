import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt


def fourier_transform(img_path):
    img = cv2.imread(img_path, 0)
    dft = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)
    dftShift = np.fft.fftshift(dft)
    result = 20 * np.log(cv2.magnitude(dftShift[:, :, 0], dftShift[:, :, 1]))

    plt.imshow(result, cmap='gray')
    plt.title('fft result')
    plt.axis('off')
    plt.savefig("./output/fourier_transform.jpg")


'''
输入：一张图片
输出：傅里叶变换的结果
'''
if __name__ == '__main__':
    fourier_transform(sys.argv[1])
