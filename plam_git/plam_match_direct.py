import cv2
import function
import numpy as np
from matplotlib import pyplot as plt

# Score_direct_Between_class = []
# Score_direct_within_class = []
# for i in range(1,31):
#     for j in range(1,31):
#         if i >= j:
#             # print(i,j)
#             continue
#         else:
#             image = cv2.imread("./palm_Gabor_binary/565_{}.bmp".format(i),cv2.COLOR_BGR2GRAY)
#             target = cv2.imread("./palm_Gabor_binary/566_{}.bmp".format(j),cv2.COLOR_BGR2GRAY)
#             # 裁剪两图至大小一致
#             # 裁剪原则 中心不变，从边缘开始对称裁剪
            
#             if target.shape[0] > image.shape[0]:
#                 err = ((target.shape[0] - image.shape[0]) >> 1) 
#                 target = target[err:(target.shape[0]-err),err:(target.shape[0]-err)]
#                 # print(image.shape,target.shape)
#             else:
#                 err = ((image.shape[0] - target.shape[0]) >> 1) 
#                 image = image[err:(image.shape[0]-err),err:(image.shape[0]-err)]
#                 # print(image.shape,target.shape)
            
#             # 直接模板匹配
#             score_direct_match ,min_value,max_value = function.match_template(image,target)
#             Score_direct_within_class.append(score_direct_match)
#             print("%.3f"%(score_direct_match * 100),'%')
# np.save(".\\score_direct_match_between_class.npy",np.array(Score_direct_within_class))


score_direct_match_between_class = np.load("score_direct_match_between_class.npy")
score_direct_match_within_class = np.load("score_direct_match_within_class.npy")

score_direct_match_between_class = np.unique(score_direct_match_between_class)
score_direct_match_within_class = np.unique(score_direct_match_within_class)

import seaborn as sns 
sns.distplot(score_direct_match_between_class,color="r",bins=30,label="Within_Class")
# plt.legend()
plt.savefig("direct类间")
plt.close()

import seaborn as sns 
sns.distplot(score_direct_match_within_class,color="skyblue",bins=30,label="Between_Class")
# plt.legend()
plt.savefig("direct类内")
plt.close()

import seaborn as sns 
sns.distplot(score_direct_match_between_class,color="r",bins=30,label="Within_Class")
sns.distplot(score_direct_match_within_class,color="skyblue",bins=30,label="Between_Class")
plt.savefig("direct类间类内")
plt.close()


"""err = 15
# 裁剪的边缘大小
target = target[err:(target.shape[0]-err),err:(target.shape[0]-err)]
print(image.shape,target.shape)

# 模板匹配_平移卷积
res = function.match_cv2_temlate(target,image)
plt.imshow(res, cmap='gray')
plt.title('Detected Point'), plt.xticks([]), plt.yticks([])
plt.show()

# 绘制彩图
result = np.zeros(res.shape, dtype=np.float32) # 定义归一化的输出矩阵 需要与输入的图保持大小一致
cv2.normalize(res, result, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
function.matshow_3d(result)"""

print("end")