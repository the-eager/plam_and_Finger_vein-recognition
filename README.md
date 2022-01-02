# plam_and_Finger_vein-recognition
## SCUT vein-recognition homework
plam_match.py 调用实现图像匹配

palm_roi_enhance.py 调用实现图像滤波与增强

function.py 主要函数  （还有其他的一些但最后没有用上的图像滤波的函数实现/我觉得他们效果一般，所以没有用）

实现的流程1  局部二进制模式： roi-clache-retinex-gabor（在function.py）-lbp（plam_match.py） 

实现的流程2  二值纹理特征： roi-clache-retinex-gabor（在function.py）- function.match_template（直接模板匹配_平移卷积 ）

局部不变特征提取 ：SIFT、RootSIFT、SURF 我只看了算法原理 https://blog.csdn.net/qq_30815237/article/details/86544234 这个系列，没有实现
