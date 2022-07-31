import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt


def template_matching(img_path, tmp_path):
    img = cv2.imread(img_path)  # 读取图片
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_part = cv2.imread(tmp_path, 0)  # 读取图片

    w, h = img_part.shape[::-1]
    # 使用cv2.TM_CCOEFF_NORMED相关性系数匹配方法时，最佳匹配结果在值等于1处，最差匹配结果在值等于-1处，值等于0直接表示二者不相关。
    res = cv2.matchTemplate(img_gray, img_part, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    top_left = max_loc  # 最大值即为最佳匹配点
    bottom_right = (top_left[0] + w, top_left[1] + h)
    cv2.rectangle(img, top_left, bottom_right, 255, 2)  # 绘制矩形
    cv2.imwrite('./output/template_matching.jpg', img)


'''
输入：目标图片，匹配模板图片
输出：在目标图片标注出模版图片出现的位置
'''
if __name__ == '__main__':
    template_matching(sys.argv[1], sys.argv[2])
