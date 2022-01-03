import numpy as np
from matplotlib import pyplot as plt
import function


# LBP 
"""
将所有图片的图像存储　palm_LBP
其直方图存储　palm_LBP_hist
直方图数据存储　feature_56x.npy
跑一次有数据了就可以注释掉了
"""

# feature = []
# for i in range(1,31):
#     image = cv2.imread("./palm_Gabor/566_{}.bmp".format(i))
#     # 取灰度图 
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#     # LBP特征提取
#     dst_image = function.LBP_Cal(image)
#     function.mkdir(".\palm_LBP\\")
#     cv2.imwrite("./palm_LBP/566_{}.jpg".format(i),dst_image)
    
#     # 计算直方图
#     n, bins, patches = plt.hist(dst_image.ravel(), bins=10, rwidth=0.8, range=(0, 255),density=True, stacked=True)
#     # n返回每个bin里元素的数量；bins返回每个bin的区间范围；patches返回每个bin里面包含的数据，是一个list
#     feature.append(n)
#     function.mkdir(".\palm_LBP_hist\\")
#     plt.savefig("./palm_LBP_hist/566_{}.jpg".format(i))
#     plt.close()
# np.save(".\\feature_566.npy",np.array(feature))
# # 跑一次有数据了就可以注释掉了,注释到这里

feature_565 = np.load("feature_565.npy")
feature_566 = np.load("feature_566.npy")

Score_LBP_Between_class = []
Score_LBP_within_class = []
for i in range(0,30):
    for j in range(0,30):
        score_LBP = function.hist_match(feature_565[int(i)],feature_566[int(j)])
        Score_LBP_Between_class.append(score_LBP)
        print("score is %.7f"%(score_LBP*100),'%')


for i in range(0,30):
    for j in range(0,30):
        score_LBP = function.hist_match(feature_565[int(i)],feature_565[int(j)])
        Score_LBP_within_class.append(score_LBP)
        print("score is %.7f"%(score_LBP*100),'%')
Score_LBP_within_class = np.unique(Score_LBP_within_class)
# 删除重复元素 100%的分数
np.save(".\\score_Within.npy",np.array(Score_LBP_within_class))

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