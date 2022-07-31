import sys

import numpy as np
import cv2

import py2og


def judge(w0, h0, w, h):
    a = img[h0: h0 + h, w0: w0 + w]
    ave = np.mean(a)
    std = np.std(a, ddof=1)
    count = 0
    total = 0
    for i in range(w0, w0 + w):
        for j in range(h0, h0 + h):
            if abs(img[j, i] - ave) < 1 * std:
                count += 1
            total += 1
    if (count / total) < 0.95:
        return True
    else:
        return False


def draw(w0, h0, w, h):
    for i in range(w0, w0 + w):
        for j in range(h0, h0 + h):
            if img[j, i] > 125:
                img[j, i] = 255
            else:
                img[j, i] = 0


def function(w0, h0, w, h):
    if judge(w0, h0, w, h) and (min(w, h) > 5):
        function(w0, h0, int(w / 2), int(h / 2))
        function(w0 + int(w / 2), h0, int(w / 2), int(h / 2))
        function(w0, h0 + int(h / 2), int(w / 2), int(h / 2))
        function(w0 + int(w / 2), h0 + int(h / 2), int(w / 2), int(h / 2))
    else:
        draw(w0, h0, w, h)

'''
B3032
argv[1]: img name
argv[2]: sid
'''
if __name__ == "__main__":
    img = cv2.imread(py2og.inputFilePath+sys.argv[1], 0)
    img_input = cv2.imread(py2og.inputFilePath+sys.argv[1], 0)#备份

    height, width = img.shape

    function(0, 0, width, height)

    # cv2.imwrite("./output/split_and_merge.jpg",img)
    cv2.imwrite(py2og.outputFilePath+sys.argv[2]+'_out.jpg', img)
    py2og.updateDBwithFinishSignal(sys.argv[2])


