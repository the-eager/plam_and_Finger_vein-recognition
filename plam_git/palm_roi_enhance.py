
import cv2
import numpy as np
from matplotlib import pyplot as plt
import function
from function import show_picture

for i in range(1,31):
    image = cv2.imread("./palm_data/566/566_{}.bmp".format(i))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # 取灰度图 

    th1,ret = function.OTSU(gray)
    # 二值化
    function.mkdir(".\palm_OTSU\\")
    cv2.imwrite("./palm_OTSU/566_{}.bmp".format(i), th1)
    # show_picture(th1)

    image, roi_img = function.roi_plam(image, th1)
    # 提取手掌ROI
    function.mkdir(".\palm_roi_with_symbol\\")
    cv2.imwrite("./palm_roi_with_symbol/566_{}.bmp".format(i), image)
    function.mkdir(".\palm_roi\\")
    cv2.imwrite("./palm_roi/566_{}.bmp".format(i), roi_img)


    '''
    图像增强
    '''
    equalizeHist_img_rgb = function.equalizeHist_for_rgbpicture_rgb(roi_img)
    # 全局直方图均衡化
    # show_picture(equalizeHist_img_rgb)
    function.mkdir(".\palm_equalizeHist\\")
    cv2.imwrite("./palm_equalizeHist/566_{}.bmp".format(i), equalizeHist_img_rgb)

    equalizeHist_img_rgb_CLAHE = function.equalizeHist_for_rgbpicture_CLAHE(roi_img)
    # 自适应直方图均衡化(CLAHE)
    # show_picture(equalizeHist_img_rgb_CLAHE)
    function.mkdir(".\palm_equalizeHist_CLAHE\\")
    cv2.imwrite("./palm_equalizeHist_CLAHE/566_{}.bmp".format(i), equalizeHist_img_rgb_CLAHE)

    Gray_World_Algorithm_img_rgb = function.Gray_World_Algorithm(roi_img)
    # 灰度世界算法
    # show_picture(Gray_World_Algorithm_img_rgb)
    function.mkdir(".\palm_Gray_World_Algorithm\\")
    cv2.imwrite("./palm_Gray_World_Algorithm/566_{}.bmp".format(i), Gray_World_Algorithm_img_rgb)

    Automatic_White_Balance_img_rgb = function.Automatic_White_Balance(roi_img)
    # 自动白平衡算法
    # show_picture(Automatic_White_Balance_img_rgb)
    function.mkdir(".\palm_Automatic_White_Balance\\")
    cv2.imwrite("./palm_Automatic_White_Balance/566_{}.bmp".format(i), Automatic_White_Balance_img_rgb)

    # function.Compared_with_oiginal_img(roi_img, equalizeHist_img_rgb )
    # function.Compared_with_oiginal_img(roi_img, equalizeHist_img_rgb_CLAHE)
    # function.Compared_with_oiginal_img(roi_img, Gray_World_Algorithm_img_rgb)
    # function.Compared_with_oiginal_img(roi_img, Automatic_White_Balance_img_rgb)

    # Retinex算法
    sigma = 30
    ## 指定尺度（模糊的半径）
    dy = 2
    #Dynamic取值越小，图像的对比度越强。
    #一般来说Dynamic取值2-3之间能取得较为明显的增强效果
    retinex_msrcr = function.msrcr(equalizeHist_img_rgb_CLAHE, sigma,dy)
    cv2.normalize(retinex_msrcr, retinex_msrcr, 0, 255, cv2.NORM_MINMAX)
    Retinex_img_rgb = cv2.convertScaleAbs(retinex_msrcr)
    # function.Compared_with_oiginal_img(roi_img, Retinex_img_rgb)
    function.mkdir(".\palm_Retinex\\")
    cv2.imwrite("./palm_Retinex/566_{}.bmp".format(i), Retinex_img_rgb)

    # Gabor滤波
    filters = function.build_filters()
    Gabor_img = function.Gabor_process(Retinex_img_rgb, filters)
    # function.Compared_with_oiginal_img(roi_img, Gabor_img)
    # function.Compared_with_oiginal_img(Retinex_img_rgb, Gabor_img)
    function.mkdir(".\palm_Gabor\\")
    cv2.imwrite("./palm_Gabor/566_{}.bmp".format(i), Gabor_img)

    Gabor_img_gray = cv2.cvtColor(Gabor_img, cv2.COLOR_BGR2GRAY)
    # OTSU 找阈值
    th_OTSU,ret_OTSU = function.OTSU(Gabor_img_gray)
    # 阈值化
    print("ret_OTSU",ret_OTSU)
    # inverse 阈值化
    ret, binary = cv2.threshold(Gabor_img_gray, ret_OTSU, 255, cv2.THRESH_BINARY_INV)
    # 腐蚀
    # binary = function.Erosion(binary)
    # show_picture(binary)
    function.mkdir(".\palm_Gabor_binary\\")
    cv2.imwrite("./palm_Gabor_binary/566_{}.bmp".format(i), binary)

    ret, binary = cv2.threshold(binary, ret_OTSU, 255, cv2.THRESH_BINARY_INV)
    # show_picture(binary)

    print(binary.shape)

print("end")
