# -*- coding: utf-8 -*-
"""
Created on Fri Oct 25 22:01:33 2019

@author: Administrator
"""



#调包
import re  #正则表达式块
import numpy   #科学计算
from numpy import *
from sklearn import linear_model   #
from matplotlib import pyplot as plt
#导入数据
fn=open(r'E:/Anaconda3/Python_work/air3.txt')
all_data=fn.readlines()
fn.close()
XH=[]
OZ=[]
RAD=[]
TEMP=[]
WIND=[]#创建空列表
#数据预处理
for single_data in all_data:
   tmp_data=re.split('\t|\n',single_data)#|或
   OZ.append(float(tmp_data[1]))  #tmp_data的第一列赋值给OZ
   RAD.append(float(tmp_data[2]))
   TEMP.append(float(tmp_data[3]))
   WIND.append(float(tmp_data[4]))
   break
OZ=numpy.array(OZ).reshape([111,1])
WIND=numpy.array(WIND).reshape([111,1])
OZ
WIND
#数据分析展示
plt.scatter(WIND,OZ)
plt.show()
#数据建模
model=linear_model.LinearRegression()
model.fit(WIND,OZ)
#模型评估
model_coef=model.coef_              #斜率
model_intercept=model.intercept_    #截距
r2=model.score(WIND,OZ)            #残差平方和
#预测
new_x=84    



#数据结构类型
random.rand(4,4)   #生成随机数数组，需要调用numpy包里的random模块
#import numpy import * 导入numpy的所有模块
#mat()函数可以将数组转化成矩阵
randMat=mat(random.rand(4,4))   #matrix为矩阵形式的数据
randMat.I   #.I对数据求逆
invRandMat=randMat.I    #存储逆运算

randMat.I *randMat     #矩阵与逆矩阵相乘，python存在一定的偏差
#计算偏差
myEye=randMat.I *randMat  
myEye-eye(4)        #eye(4)生成4*4的单位矩阵

from numpy import *   #科学计算包
import operator   #运算符模块
distances=tile([5,1,2,4],1)
distances.argsort() 


