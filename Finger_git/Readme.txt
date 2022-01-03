# 数据格式解释 001_1到001_6是001的第一根手指
# 001_7到001_12是001的第二根手指 001一共有24张图（4根手指*每根6只）
# 002_1到002_24是第二个人的24张手指图
# 003的手指关节不全，4根手指只有2根，直接删除不用了
# 类内距的解释 只有同个人的同个手指才算类间 （同类样本就6张）
# 类间距的解释 其余都算不同类 

plam_match.py 调用实现图像匹配
palm_roi_enhance.py 调用实现图像滤波与增强
function.py 主要函数  （还有其他的一些最后没有用上的图像滤波的函数实现/我觉得他们效果一般，所以没有用）
实现的流程1  局部二进制模式： roi-clache-retinex-gabor（在function.py）-lbp（plam_match.py） 
实现的流程2  二值纹理特征： roi-clache-retinex-gabor（在function.py）- function.match_template（直接模板匹配_平移卷积 ）
局部不变特征提取 ：SIFT、RootSIFT、SURF 都有专利保护 需要cv库降版本躲专利限制  我只看了算法原理 https://blog.csdn.net/qq_30815237/article/details/86544234 这个系列，没有实现
"""
练手作业，如果可以，要不给个Star,谢谢啦  o(*≧▽≦)ツ
欢迎校友交流 By松仔松仔松仔松
"""