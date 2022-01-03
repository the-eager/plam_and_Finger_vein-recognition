import numpy as np
from matplotlib import pyplot as plt
import function
import cv2

# LBP 
"""
将所有图片的图像存储 palm_LBP
其直方图存储 palm_LBP_hist
直方图数据存储 feature_56x.npy
跑一次有数据了就可以注释掉了
"""

# feature = []
# for i in range(1,25):
#     image = cv2.imread("./Finger_Gabor/002_{}.bmp".format(i))
#     # 取灰度图 
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#     # LBP特征提取
#     dst_image = function.LBP_Cal(image)
#     function.mkdir(".\Finger_LBP\\")
#     cv2.imwrite("./Finger_LBP/002_{}.jpg".format(i),dst_image)
    
#     # 计算直方图
#     n, bins, patches = plt.hist(dst_image.ravel(), bins=10, rwidth=0.8, range=(0, 255),density=True, stacked=True)
#     # n返回每个bin里元素的数量；bins返回每个bin的区间范围；patches返回每个bin里面包含的数据，是一个list
#     feature.append(n)
#     function.mkdir(".\Finger_LBP_hist\\")
#     plt.savefig("./Finger_LBP_hist/002_{}.jpg".format(i))
#     plt.close()
# np.save(".\\feature_002.npy",np.array(feature))
# # 跑一次有数据了就可以注释掉了,注释到这里

feature_001 = np.load("feature_001.npy")
feature_002 = np.load("feature_002.npy")
# 数据格式解释 001_1到001_6是001的第一根手指
# 001_7到001_12是001的第二根手指 001一共有24张图（4根手指*每根6只）
# 002_1到002_24是第二个人的24张手指图
# 003的手指关节不全，4根手指只有2根，直接删除不用了
# 类内距的解释 只有同个人的同个手指才算类间 （同类样本就6张）
# 类间距的解释 其余都算不同类 
Score_LBP_Between_class = []
Score_LBP_within_class = []

for i in range(0,24):
    for j in range(0,24):
        score_LBP = function.hist_match(feature_001[int(i)],feature_002[int(j)])
        Score_LBP_Between_class.append(score_LBP)
        print("score is %.7f"%(score_LBP*100),'%')



for i in range(0,6):
    for j in range(i+1,6):
        print(i,j)
        score_LBP = function.hist_match(feature_001[int(i)],feature_001[int(j)])
        Score_LBP_within_class.append(score_LBP)
        print("score is %.7f"%(score_LBP*100),'%')



for i in range(6,12):
    for j in range(i+1,12):
        print(i,j)
        score_LBP = function.hist_match(feature_001[int(i)],feature_001[int(j)])
        Score_LBP_within_class.append(score_LBP)
        print("score is %.7f"%(score_LBP*100),'%')

for i in range(12,18):
    for j in range(i+1,18):
        print(i,j)
        score_LBP = function.hist_match(feature_001[int(i)],feature_001[int(j)])
        Score_LBP_within_class.append(score_LBP)
        print("score is %.7f"%(score_LBP*100),'%')

for i in range(18,24):
    for j in range(i+1,24):
        print(i,j)
        score_LBP = function.hist_match(feature_001[int(i)],feature_001[int(j)])
        Score_LBP_within_class.append(score_LBP)
        print("score is %.7f"%(score_LBP*100),'%')

Score_LBP_within_class = np.unique(Score_LBP_within_class)
# 删除重复元素 100%的分数


import seaborn as sns 
sns.distplot(Score_LBP_Between_class,color="r",bins=30,label="Within_Class")
# plt.legend()
plt.savefig("lbp类间")
plt.close()

import seaborn as sns 
sns.distplot(Score_LBP_within_class,color="skyblue",bins=30,label="Between_Class")
# plt.legend()
plt.savefig("lbp类内")
plt.close()

import seaborn as sns 
sns.distplot(Score_LBP_Between_class,color="r",bins=30,label="Within_Class")
sns.distplot(Score_LBP_within_class,color="skyblue",bins=30,label="Between_Class")
plt.savefig("lbp类间类内")
plt.close()


print("end")