# KNN-Arcace
## 安装依赖
### 所需要的依赖库caffe,opencv,cuda
#### 使用https://github.com/BVLC/caffe.git 的caffe，并且将caffelayer下的额外层axpylayer分别加入caffe安装包中caffe/include/caffe/layer和caffe/src/caffe/layers,然后编译caffe
#### 下载cuda,并安装
#### 下载opencv,并附带cuda安装
## 运行
### windows版本
#### 配置caffe,opencv,cuda头文件路径，依赖库路径
#### 1.建立VS项目，解决方案下项目右键->生成依赖项->生成自定义，勾选CUDA
#### 2.右键knn.cu->属性，选择CUDA
#### 3.运行项目
### linux版本
#### 1.使用nvcc编译knn.cu文件 nvcc -c knn.cu 
#### 2.g++编译arcface.cpp文件 g++ -c arcface.cpp
#### 3.g++ -o a.out knn.o arcface.o 头文件目录 库文件目录 库
#### 缺失face.caffemodel请下载
# Note
#### main函数在arcface中，并且未将头文件与库文件分离，若要使用请自行分离
#### 使用Arcface提取512维特征，并且使用knn进行类别分离
# 已训练模型下载
### https://pan.baidu.com/s/11qT4DRXIIfHVkoZdNZwOPQ 获取人脸识别模型
### 其余模型都在model文件中
### 其余更小更快的mobileNet请查看arcfacemodel
# 参考
### 基于ResNet的Arcface https://github.com/deepinsight/insightface
### MTCNN的特征点检测检测模型ONet https://github.com/blankWorld/MTCNN-Accelerate-Onet
