import cv2 
import function
from matplotlib import pyplot as plt
for k in range(1,7):
    clahe_test=cv2.imread("./palm_data/finger/002_4/0{}.jpg".format(k))
    image = cv2.cvtColor(clahe_test, cv2.COLOR_BGR2GRAY)
    # k = k+18
    # k = 1-6 是001文件夹的01六张图 / k = 7-12 是001文件夹的02的六张图
    # 边缘检测
    sobelx = cv2.Sobel(image,cv2.CV_64F,1,0,ksize=5)
    canny=cv2.Canny(image,90,150)
    
    function.mkdir(".\Finger_clahe_test\\")
    cv2.imwrite("./finger_clahe_test/002_{}.bmp".format(k), clahe_test)

    function.mkdir(".\Finger_sobelx\\")
    cv2.imwrite("./finger_sobelx/002_{}.bmp".format(k), sobelx)

    function.mkdir(".\Finger_canny\\")
    cv2.imwrite("./finger_canny/002_{}.bmp".format(k), canny)





    th1,ret = function.OTSU(image)
    # 二值化
    function.mkdir(".\Finger_OTSU\\")
    cv2.imwrite("./Finger_OTSU/002_{}.bmp".format(k), th1)
    # function.show_picture(th1)

    import math
    roi_img = clahe_test.copy()
    # 1.先画出一个圆k
    distance = cv2.distanceTransform(th1, cv2.DIST_L2, 5, cv2.CV_32F)
    # th1 是阈值化后的图
    # Calculates the distance to the closest zero pixel for each pixel of the source image.
    maxdist = 0
    # rows,cols = img.shape
    for i in range(distance.shape[0]):
        for j in range(distance.shape[1]):
            dist = distance[i][j]
            if maxdist < dist:
                x = j
                y = i
                maxdist = dist
    # 思路是遍历找一个点 让这个点距离其他黑色部分（背景）的距离最大
    # print(x, y)
 
    (left, right, top, bottom) = ((x - 50),(x + 50), (y - 200), (y + 150))
    
    if top < 0:
        top = 0
    if bottom > roi_img.shape[0]-1:
        bottom = roi_img.shape[0]-1
    p1 = (int(left), int(top))
    p2 = (int(right), int(bottom))
    cv2.rectangle(image, p1, p2, (255, 0, 0), 1, 1)
    # function.show_picture(image)
    function.mkdir(".\Finger_roi_with_symbol\\")
    cv2.imwrite("./Finger_roi_with_symbol/002_{}.bmp".format(k), image)

    roi_img = roi_img[int(top):int(bottom), int(left):int(right)]
    function.mkdir(".\Finger_roi\\")
    cv2.imwrite("./Finger_roi/002_{}.bmp".format(k), roi_img)
    


    equalizeHist_img_rgb_CLAHE = function.equalizeHist_for_rgbpicture_CLAHE(roi_img)
    # 自适应直方图均衡化(CLAHE)
    # show_picture(equalizeHist_img_rgb_CLAHE)
    function.mkdir(".\Finger_equalizeHist_CLAHE\\")
    cv2.imwrite("./Finger_equalizeHist_CLAHE/002_{}.bmp".format(k), equalizeHist_img_rgb_CLAHE)

    # Retinex算法
    sigma = 30
    ## 指定尺度（模糊的半径）
    dy = 2
    #Dynamic取值越小，图像的对比度越强。
    #一般来说Dynamic取值2-3之间能取得较为明显的增强效果
    retinex_msrcr = function.msrcr(equalizeHist_img_rgb_CLAHE, sigma,dy)
    cv2.normalize(retinex_msrcr, retinex_msrcr, 0, 255, cv2.NORM_MINMAX)
    Retinex_img_rgb = cv2.convertScaleAbs(retinex_msrcr)
    function.mkdir(".\Finger_Retinex\\")
    cv2.imwrite("./Finger_Retinex/002_{}.bmp".format(k), Retinex_img_rgb)

    # Gabor滤波
    filters = function.build_filters()
    Gabor_img = function.Gabor_process(Retinex_img_rgb, filters)
    function.mkdir(".\Finger_Gabor\\")
    cv2.imwrite("./Finger_Gabor/002_{}.bmp".format(k), Gabor_img)

    Gabor_img_gray = cv2.cvtColor(Gabor_img, cv2.COLOR_BGR2GRAY)
    # OTSU 找阈值
    th_OTSU,ret_OTSU = function.OTSU(Gabor_img_gray)
    # 阈值化
    print("ret_OTSU",ret_OTSU)
    # inverse 阈值化
    ret, binary = cv2.threshold(Gabor_img_gray, ret_OTSU, 255, cv2.THRESH_BINARY_INV)
    # 顺时针旋转
    binary = function.rotate(binary)
    function.mkdir(".\Finger_Gabor_binary\\")
    cv2.imwrite("./Finger_Gabor_binary/002_{}.bmp".format(k), binary)

    
print("end")