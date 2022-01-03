
import cv2
import numpy as np
from matplotlib import pyplot as plt
"""
练手作业，如果可以，要不给个Star,谢谢啦  o(*≧▽≦)ツ
欢迎校友交流 By松仔松仔松仔松
"""

# 绘制图像的直方图 / 用于直方图均衡化 
def hist_show(img):
    plt.hist(img.ravel(), bins=10, rwidth=0.8, range=(0, 255))
    # .ravel() .flatten() np数组扁平化 后者同步变化
    plt.show()
    return 

# 二值化
def OTSU(gray):
    '''
    输入 gray 灰度图 cv2.COLOR_BGR2GRAY 读取保存阈值化后的图到 ./OTSU/OTSU_565_{}.png'.format(i)
    输出 阈值化后的图像
    将图像高级的阈值分割 区分前景和后景
    https://www.cnblogs.com/april0315/p/13576778.html
    '''
    ret, th1 = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)  # 方法选择为THRESH_OTSU
    '''
    threshold 阈值分割
    (原始图，用于对像素值进行分类，分配给超过阈值的像素值的最大值，类型）
    ret1 = 阈值
    th = 图片
    '''
    plt.imshow(th1, "gray")
    plt.title("OTSU,threshold is " + str(ret)), plt.xticks([]), plt.yticks([])
    plt.close()
    # plt.savefig('./OTSU/OTSU_565_{}.png'.format(i))
    return th1,ret

# 手掌roi
def roi_plam(image, th1):
    '''
    输入 image 手掌原图  th1 阈值化后图原图
    输出 output roi_手掌和原图
    截取手掌的掌心部分
    '''
    import math
    roi_img = image.copy()
    # 1.先画出一个圆
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
    print(x, y)
    roi = cv2.circle(image, (x, y), math.ceil(maxdist), (255, 100, 255), 1, 8, 0)
    '''
    图像展示画好手掌的内切圆
    cv2.imshow('roi',roi)
    cv2.waitKey()
    cv2.destroyAllWindows()
    '''

    # 2.开始画圆的内接正方形
    half_slide = maxdist * math.cos(math.pi / 4)
    (left, right, top, bottom) = ((x - half_slide),
                                  (x + half_slide), (y - half_slide), (y + half_slide))
    p1 = (int(left), int(top))
    p2 = (int(right), int(bottom))
    cv2.rectangle(image, p1, p2, (77, 255, 9), 1, 1)
    roi_img = roi_img[int(top):int(bottom), int(left):int(right)]

    return image, roi_img

# 展示图片
def show_picture(image_for_show):
    '''
    输入 图
    输出 无
    显示图像大小并show出来
    '''
    cv2.imshow('image_for_show', image_for_show)
    print(image_for_show.shape)
    cv2.waitKey()
    cv2.destroyAllWindows()
    return

# 全局直方图均衡
def equalizeHist_for_rgbpicture_rgb(img_for_equalizeHist):
    '''
    输入 img_for_equalizeHist 待图像均衡化的彩色图像 原图 （对应 roi_img） 输入是rgb图
    输出 rgb图 
    直方图均衡化是使图像直方图变得平坦的操作。直方图均衡化能够有效地解决图像整体过暗、过亮的问题，增加图像的清晰度
    https://blog.csdn.net/missyougoon/article/details/81632166
    https://blog.csdn.net/Ibelievesunshine/article/details/104922449
    原始图像的灰度直方图从比较集中的某个灰度区间变成在全部灰度范围内的均匀分布
    缺点： 
 　 1）变换后图像的灰度级减少，某些细节消失； 
 　 2）某些图像，如直方图有高峰，经处理后对比度不自然的过分增强。
    https://blog.csdn.net/jyt1129/article/details/63685165
    '''
    (b, g, r) = cv2.split(img_for_equalizeHist)
    bH = cv2.equalizeHist(b)
    gH = cv2.equalizeHist(g)
    rH = cv2.equalizeHist(r)
    # 合并每一个通道
    result = cv2.merge((bH, gH, rH))
    return result

# 自适应局部直方图均衡
def equalizeHist_for_rgbpicture_CLAHE(img):
    '''
    输入 img rgb原图
    输出 enhanced_img 增强后的rgb图
    自适应直方图均衡化(AHE)用来提升图像的对比度的一种计算机图像处理技术。
    和普通的直方图均衡算法不同，AHE算法通过计算图像的局部直方图，然后重新分布亮度来来改变图像对比度。
    '''
    # 直接对 RBG 图像进行自适应局部直方图均衡化，对每个通道进行均衡化
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)) # create a CLAHE object (Arguments are optional).
    (B, G, R) = cv2.split(img)
    R, G, B = clahe.apply(R), clahe.apply(G), clahe.apply(B)
    enhanced_img = cv2.merge([B,G,R])
    # cv2.imshow('Enhanced Color Image',np.hstack([img,enhanced_img]))
    # cv2.waitKey(0)
    # cv2.destroyWindow('Enhanced Color Image')
    # cv2.imwrite('result4.png',np.hstack([img,enhanced_img]))
    return enhanced_img

# 灰度世界算法 调节白平衡
def Gray_World_Algorithm(img):
    '''
    输入 img rgb原图
    输出 Gray World Algorithm 图像增强
    该假设认为：对于一幅有着大量色彩变化的图像，三个分量的平均值趋于同一灰度值。
    自然界景物对于光线的平均反射的均值在总体上是个定值，这个定值近似地为“灰色”。
    灰度世界算法将这一假设强制应用于待处理图像，可以从图像中消除环境光的影响，获得原始场景图像。
    一般有两种方法确定Gray值：
    (1) 使用固定值，对于8位的图像(0~255)通常取128作为灰度值 Gray
    (2) 计算增益系数,分别计算三通道的平均值 avgR，avgG，avgB，则：Gray=(avgR+avgG+avgB)/3
    接着，计算增益系数 kr=Gray/avgR , kg=Gray/avgG , kb=Gray/avgB。利用计算出的增益系数，重新计算每个像素值，构成新的图片
    https://blog.csdn.net/Code_Mart/article/details/97918174
    '''
    # 取三通道的平均值作为灰度值
    avgB = np.average(img[:, :, 0])  
    avgG = np.average(img[:, :, 1])  
    avgR = np.average(img[:, :, 2])  
    avg = (avgB + avgG + avgR) / 3

    result = np.zeros(img.shape,dtype=np.uint8)
    result[:, :, 0] = np.minimum(img[:, :, 0] * (avg / avgB), 255)
    result[:, :, 1] = np.minimum(img[:, :, 1] * (avg / avgG), 255)
    result[:, :, 2] = np.minimum(img[:, :, 2] * (avg / avgR), 255)
    return result

# 自动白平衡调节算法
def Automatic_White_Balance(img):
    '''
    输入 原图rgb
    输出 图像增强后的 自动白平衡 rgb 图
    假设图像中 R, G, B 最高灰度值对应于图像中的白点，
    最低灰度值的对应于图像中最暗的点；其余像素点利用 (ax+b) 映射函数把彩色图像中 R, G, B 三个通道内的像素灰度值映射到[0.255]的范围内。
    '''
    result = cv2.cvtColor(img,cv2.COLOR_BGR2LAB)
    avg_a = np.average(result[:, :, 1])
    avg_b = np.average(result[:, :, 2])

    for x in range(result.shape[0]):
        for y in range(result.shape[1]):
            l, a, b = result[x, y, :]
            l*=100/255.0
            result[x,y,1] = a - (avg_a-128)*(1/100.0)*1.1
            result[x,y,2] = a - (avg_b-128)*(1/100.0)*1.1
    result = cv2.cvtColor(result,cv2.COLOR_LAB2BGR)
    return result

# 两图对比展示
def Compared_with_oiginal_img(original_img,result):   
    cv2.imshow('Compared',np.hstack([original_img,result]))
    cv2.waitKey(0)
    cv2.destroyWindow('Compared')
    # cv2.imwrite(str(name),np.hstack([original_img,result]))
    return 

# single scale retinex
def ssr(img, sigma):
    '''
    Retinex算法所做的就是合理地估计图像中各个位置的噪声，并除去它
    https://blog.csdn.net/wsp_1138886114/article/details/83096109
    https://juejin.cn/post/6844903647231344653
    https://github.com/JanetLau0310/Medical-Image-Enhancement/blob/master/method_retinex.ipynb
    '''
    temp = cv2.GaussianBlur(img, (0,0), sigma)
    # 叠加一个高斯噪声
    gaussian = np.where(temp==0, 0.01, temp)
    # 当某处像素值为0时，将该处值变为0.01，其余部分不变
    img_ssr = np.log10(img+0.01) - np.log10(gaussian)
    # 亮度图像 = log10原始图像 - log10叠加入射光的图像
    # https://blog.csdn.net/ajianyingxiaoqinghan/article/details/71435098
    return img_ssr

# 带颜色恢复的MSR方法MSRCR
def msrcr(img, sigma,dynamic):
    '''
    带颜色恢复的MSR方法MSRCR(Multi-Scale Retinex with Color Restoration)
    局部对比度提高，亮度与真实场景相似
    '''
    img_msrcr = np.zeros_like(img*1.0)
    img = ssr(img,sigma)
    
    img_arr = img
    mean = np.mean(img_arr)
    sum1 = img_arr.sum()
    img_arr2 = img_arr * img_arr
    sum2 = img_arr2.sum()
    var_squ = sum2 - 2*mean*sum1 + 1024*1024*mean*mean
    var = np.sqrt(var_squ)    
    Min = mean - dynamic*var
    Max = mean + dynamic*var
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            for k in range(img.shape[2]):
                img_msrcr[i][j][k] = (img[i][j][k] - Min) / (Max-Min)*255
                # 溢出判断
                # 如果有溢出将值还原回[0,255]范围中
                if img_msrcr[i][j][k] > 255:
                    img_msrcr[i][j][k] = 255
                if img_msrcr[i][j][k] < 0:
                    img_msrcr[i][j][k] = 0    
    return img_msrcr

# Gaborl 滤波
def build_filters():
    """ returns a list of kernels in several orientations
    """
    filters = []
    ksize = 31  #gaborl尺度 这里是一个
    for theta in np.arange(np.pi / 6, np.pi, np.pi / 6):   #gaborl方向 0 45 90 135 角度尺度的不同会导致滤波后图像不同        
        params = {'ksize':(ksize, ksize), 'sigma':3.3, 'theta':theta, 'lambd':18.3,    
                  'gamma':4.5, 'psi':0.89, 'ktype':cv2.CV_32F}
        #gamma越大核函数图像越小，条纹数不变，sigma越大 条纹和图像都越大
        #psi这里接近0度以白条纹为中心，180度时以黑条纹为中心
        #theta代表条纹旋转角度
        #lambd为波长 波长越大 条纹越大
        kern = cv2.getGaborKernel(**params)  #创建内核
        '''
        输入 img rgb图
        输出 滤波后的图
        Gabor滤波器可以在频域上不同尺度、不同方向上提取相关的特征
        Ksize: 滤波器的尺寸
        Sigma : 高斯函数的标准差
        Theta: 高斯函数
        Lambd: 余弦函数的波长
        Gamma: 高斯函数的宽高比，因为是二维高斯函数
        Psi： 余弦函数的相位
        '''
        kern /= 1.5*kern.sum()
        filters.append((kern,params))
    return filters                                                          

# 滤波过程
def Gabor_process(img, filters):
    """ 
    returns the img filtered by the filter list
    """
    accum = np.zeros_like(img)   #初始化img一样大小的矩阵
    for kern,params in filters:
        fimg = cv2.filter2D(img, cv2.CV_8UC3, kern)   #2D滤波函数  kern为其滤波模板
        np.maximum(accum, fimg, accum)  #参数1与参数2逐位比较  取大者存入参数3  这里就是将纹理特征显化更加明显
    return accum

# 检验并创建文件夹
import os
def mkdir(path):
    # 去除首位空格
    path=path.strip()
    # 去除尾部 \ 符号
    path=path.rstrip("\\")
    isExists=os.path.exists(path)
    if not isExists:
        # 如果不存在则创建目录
        # 创建目录操作函数
        os.makedirs(path) 
        print (path +' 创建成功')
        return True
    else:
        # 如果目录存在则不创建，并提示目录已存在
        print (path +' 目录已存在')
        return False

# 图像腐蚀
# 实测不用比较好
def Erosion(img):
    kernel = np.ones((2,2),np.uint8)
    erosion = cv2.erode(img,kernel)
    #原图　卷积核　次数
    return erosion

# 模板匹配_定义法
import collections
def match_template(image,target):
    min_value = cv2.min(image,target) # 交集
    max_value = cv2.max(image,target) # 并集
    count_min = 0
    count_max = 0
    for i in range(0,min_value.shape[0]):
        count_min += collections.Counter(min_value[i])[255]
        count_max += collections.Counter(max_value[i])[255]
    result = float(float(count_min) / float(count_max))
    return result,min_value,max_value

#  模板匹配_平移卷积
def match_cv2_temlate(image,template):
    res = cv2.matchTemplate(image, template, cv2.TM_CCOEFF)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    print(min_val, max_val, min_loc, max_loc)
    return res

#　直方图匹配
def match_compareHist(image,target):
    #only calc grey.重置图像大小
    hist_show(image)
    hist_show(target)
    # greyx = cv2.cvtColor(imgx, cv2.COLOR_BGR2GRAY)#灰度图
    histx = cv2.calcHist([image], [0], None, [cv2.HistSize], [0.0, 256.0])#获取图像直方图数据
    histy = cv2.calcHist([target], [0], None, [cv2.HistSize], [0.0, 256.0])
    res = cv2.compareHist(histx, histy, cv2.CV_COMP_CORREL)
    #opencv中的compareHist函数是用来计算两个直方图相似度，计算的度量方法有4个，分别为Correlation ( CV_COMP_CORREL )相关性，Chi-Square ( CV_COMP_CHISQR ) 卡方，Intersection ( method=CV_COMP_INTERSECT )交集法，Bhattacharyya distance ( CV_COMP_BHATTACHARYYA )常态分布比对的Bhattacharyya距离法。
    #compareHist函数返回一个数值，相关性方法范围为0到1,1为最好匹配，卡方法和Bhattacharyya距离法是值为0最好，而交集法为值越大越好。
    return res

# 绘制3维网格图
def  Wireframe_plots(img_gray):
    Y = np.arange(0, np.shape(img_gray)[0], 1)
    X = np.arange(0, np.shape(img_gray)[1], 1)
    X, Y = np.meshgrid(X, Y) #　生成网格点坐标矩阵
    fig = plt.figure()
    ax = plt.axes(projection="3d")
    ax.plot_wireframe(X, Y, img_gray, rstride=10, cstride=10)
    plt.show()

# 绘制3维彩图
def matshow_3d(img_gray):
    Y = np.arange(0, np.shape(img_gray)[0], 1)
    X = np.arange(0, np.shape(img_gray)[1], 1)
    X, Y = np.meshgrid(X, Y) #　生成网格点坐标矩阵
    fig = plt.figure() # figure(num=None, figsize=None, dpi=None, facecolor=None, edgecolor=None, frameon=True)
    ax = plt.axes(projection="3d")
    ax.plot_surface(X, Y, img_gray, cmap=cm.gist_rainbow)
    plt.show()
    return

# LBP
def LBP_Cal(src):
    """
    输入 单通道原图
    输出 LBP特征图
    """
    dst = np.zeros(src.shape,dtype = src.dtype)
    for row in range(1,src.shape[0]-1):
        for col in range(1,src.shape[1]-1):
            center = src[row,col]
            LBPtemp = 0
            # 
            LBPtemp |= (src[row-1,col-1] >= center) << 7 
            # 左移七位 改变第一个像素点的位置 如果这个位置数字大，该位置数变为1 与原来的LBPtemp（最初是0000 0000）想与 只改变目标位置的数字
            LBPtemp |= (src[row-1,col  ] >= center) << 6
            LBPtemp |= (src[row-1,col+1] >= center) << 5
            LBPtemp |= (src[row  ,col-1] >= center) << 4
            LBPtemp |= (src[row  ,col+1] >= center) << 3
            LBPtemp |= (src[row+1,col-1] >= center) << 2
            LBPtemp |= (src[row+1,col  ] >= center) << 1
            LBPtemp |= (src[row+1,col+1] >= center) << 0

            dst[row,col] = LBPtemp

    return dst

# rgb三通道三张直方图比较 (3*3分布)
def compare_histogram_rgb(image,reference,target):
    '''
    exposure.histogram 和 np.histogram 的区别 
    分成两个bin，每个bin的统计量是一样的，但numpy返回的是每个bin的两端的范围值，而skimage返回的是每个bin的中间值
    np.histogram(image, bins=2)   #用numpy包计算直方图
    exposure.histogram(image, nbins=2)  #用skimage计算直方图
    此外还有 plt.hist
    '''
    from skimage import exposure
    fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(8, 8))
    for i, img in enumerate((image,reference,target)):
        for c, c_color in enumerate(('red', 'green', 'blue')):
            img_hist, bins = exposure.histogram(img[:,:, c], source_range='dtype')
            axes[c, i].plot(bins, img_hist / img_hist.max())
            img_cdf, bins = exposure.cumulative_distribution(img[:,:, c])
            axes[c, i].plot(bins, img_cdf)
            axes[c, 0].set_ylabel(c_color)
    axes[0, 0].set_title('Source')
    axes[0, 1].set_title('Reference')
    axes[0, 2].set_title('Matched')
    plt.tight_layout()
    plt.show()
    return 

# 直方图匹配 输出直方图分数
def hist_match(hist_template,hist_test):
    '''
    输入 两张直方图每个bin里面的数量 n =  plt.hist()
    输出 分数 建议搭配 print("score is %.2f"%(score_LBP*100),'%')
    ''' 
    template_sum=hist_template.sum()
    hist_test_sum=hist_test.sum()
    result=0
    for i in range(hist_template.size):
        if hist_template[i]<=hist_test[i]:
            result=result+hist_template[i]
        else:
            result=result+hist_test[i]
    if template_sum >= hist_test_sum:
        sum = template_sum
    else:
        sum = hist_test_sum
    score=result/sum
    return score

# 单张图的直方图和累计曲线计算与输出
from skimage import exposure # 图像处理包
def hist(dst_image):
    # 计算直方图用于直方图匹配 
    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(5, 5),squeeze=False)
    img_hist, bins = exposure.histogram(dst_image, source_range='dtype',nbins=255,normalize=True)
    #　直方图
    axes[0,0].plot(bins, img_hist / img_hist.max())
    img_cdf, bins = exposure.cumulative_distribution(dst_image)
    # 累计曲线图
    axes[0,0].plot(bins, img_cdf)
    return hist

# 这个是分割成 2*4 格的LBP 这里我没有用,因为不分割同样效果很好
# 这里的 LBP_hist 和 normalization 都是用于学习参考
from skimage import feature
def LBP_hist(ROI_img=None):  
    img=feature.local_binary_pattern(ROI_img,8,3,'default')
    img=img.astype('uint8')
    tile_cols=4
    tile_rows=2
    n=1
    rows,cols=img.shape
    step_rows=round(rows/tile_rows)-1
    step_cols=round(cols/tile_cols)-1
    feature1=[]
    for i in range(tile_rows):
        for j in range(tile_cols):
            tile=img[i*step_rows:(i+1)*step_rows-1,j*step_cols:(j+1)*step_cols-1]
            hist=cv2.calcHist([tile],[0],None,[255],[0,255])
            hist=normalization(hist)
            feature1.append(hist) 
            plt.subplot(240+n)
            plt.plot(hist)
            n+=1
    plt.show()
    feature1=np.array(feature1)
    feature1=feature1.ravel()
    return feature1

def normalization(array) : #array 为nparry
    array=array.ravel()
    result =[]
    array_sum=array.sum()
    for i in array:
        result.append(i/array_sum)
    return np.array(result)

# 图像顺时针旋转90度
def rotate(img_src):
    height, width = img_src.shape[:2]
    # 创建X,Y map
    map_x = np.zeros([width, height], np.float32)
    map_y = np.zeros([width, height], np.float32)
    # 执行重映射 调整 X Y map位置
    for i in range(width):
        for j in range(height):
            map_x.itemset((i, j), i)
            map_y.itemset((i, j), j)
    # 执行重映射处理
    img_dst = cv2.remap(img_src, map_x, map_y, cv2.INTER_LINEAR)
    return img_dst
 
