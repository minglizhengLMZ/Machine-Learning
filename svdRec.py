# -*- coding: utf-8 -*-
"""
Created on Sun Dec  8 08:30:01 2019

@author: lenovo
"""
'''
SVD,奇异值分解：提取信息的强大工具

应用:
    1最早应用于信息检索（可以实现同义词搜索），隐形语易索引LSI或者LSA
    2后来应用于推荐系统（比原始数据集更好的推荐效果）
    3.图像压缩

优点：简化数据去除噪声，提高算法的结果
缺点：数据的转换可能难以理解
适用数据：数值型数据

SVD：去除噪音和冗余信息，可以看成是从有噪音数据中抽取相关特征

SVD 用于推荐系统，先从原始数据构造主题空间，然后在该空间计算相似度

分解公式：
D_data=U_{m*n}{奇异值矩阵}V.T_{m*n}
奇异值矩阵：对角线元素为奇异值（按从小到大顺序排列），其他元素为0，
奇异值为：Data*Data.T特征值的平方根

PCA：为特征值和特征向量

某个奇异值之后，其他奇异值都为0，其余特征都是噪声或者冗余特征

'''


'''------------------python实现SVD-----------------------------------------'''

from numpy import *
U,Sigma,VT=linalg.svd([[1,1],[7,7]])
U
VT
Sigma     #实质是矩阵，在运算中也是按矩阵运算，但为节省空间，仅返回对角线元素
#创建数据
def loadExData():
    return[[1,1,0,2,2],
           [2,0,0,3,3],
           [1,0,0,1,1],
           [1,1,1,0,0],
           [2,2,2,0,0],
           [1,1,1,0,0],
           [5,5,5,0,0]]

Data=loadExData()
#奇异值分解
U,Sigma,VT=linalg.svd(Data)
Sigma    #后面两个奇异值很小，可以将其去掉，
shape(U)
#Data_m*n 约等于 U_{m*3}*{去掉较小值后的奇异值矩阵}*V.T_{n*n}

#重构原始矩阵的近似矩阵（仅保留前三个奇异值），对应U的前三列，V的前三行
Sig3=mat([[Sigma[0],0,0],[0,Sigma[1],0],[0,0,Sigma[2]]])
U[:,:3]*Sig3*VT[:3,:]


#确定保留保留奇异值的数目：保留矩阵中90%的能量信息

'''------------SVD在推荐引擎中的应用-----（基于协同过滤的推荐引擎）-----------'''
'''原理用过将用户通过和其他用户对比来实现推荐
用户和该物品的相似度高，则把该物品分配给用户
协同过滤并不关心物品属性，而是严格按照许多用户的观点来计算相似度

我们希望相似度在0-1之间变化，并且物品间对越相似，他们的相似度越大

相似度计算方法：

1相似度的计算公式：相似度=1/(1+距离)，距离为两物品间的欧式距离

2.第二种计算距离的方法：皮尔逊相关系数corrcoef()（优点,对用户评级的量级并不敏感）
皮尔逊相关系数的取值在-1——1之间， 归一化变换到0-1之间  0.5+0.5*corrcoef()

余弦相似度：两向量夹角的余弦值，夹角为90°，相似度为0，夹角为0，两向量方向相同，相似度为1.0
余弦相似度的取值范围也在-1——1之间，将其规划到0——1之间，
cos角度 = A*B/(||A|| * ||B||)  分母为A，B的2范数的乘积

'''#---------------------相关性的三种度量方法-------------------------------
from numpy import *
from numpy import linalg as la       #线性代数工具箱

#用欧式距离(即向量对应元素差的2范数)计算相似度
def ecludSim(inA,inB):
    return 1.0/(1.0+la.norm(inA-inB))
#用皮尔逊相关系数计算相似度
def pearsSim(inA,inB):
    if len(inA)<3:return 1.0
    return 0.5+0.5*corrcoef(inA,inB,rowvar=0)[0][1]   #仅要计算相关系数矩阵第一行第一列的数据

#rowvar=0表示传入数据为一行为一个样本如果不指定0，表示传入数据每一列为一个元素
#[0][1]   #获取第一行第一列元素，为所求相关系数
def cosSim(inA,inB):
    num=float(inA.T*inB)     #分子
    denom=la.norm(inA)*la.norm(inB)   #分母
    return 0.5+0.5*(num/denom)

#测试，计算列向量之间的相似度，'(计算行向量就会出问题),表明基于物品的相似度计算方法
    #欧式距离计算相似度
myMat=mat(loadExData())
ecludSim(myMat[:,0],myMat[:,4])
ecludSim(myMat[:,1],myMat[:,1])
   #余弦相似度
cosSim(myMat[:,0],myMat[:,4])
cosSim(myMat[:,0],myMat[:,0])
   #皮尔逊相似度
pearsSim(myMat[:,0],myMat[:,4])
pearsSim(myMat[:,0],myMat[:,0])

'''在此例中（行与行之间的比较是基于用户之间的相似度，列于列之间的比较是基于物品的相似度）
#对于大部分商品，用户的数量会多于出售商品的种类，倾向于基于商品计算相似度

#推荐引擎的评价：
采用交叉测试的方法。【拆分数据为训练集和测试集】
推荐引擎评价的指标： 最小均方根误差(Root mean squared error, RMSE)，也称标准误差(Standard error)，就是计算均方误差的平均值然后取其平方根。
如果RMSE=1, 表示相差1个星级；如果RMSE=2.5, 表示相差2.5个星级。
'''


'''------------------示例菜肴馆的推荐引擎--（推荐未吃过的可能喜欢的菜肴）-----------------------------'''
'''
推荐系统的工作过程：给定一个用户，系统会为此用户返回N个最好的推荐菜。
实现流程大致如下：
寻找用户没有评级的菜肴，即在用户-物品矩阵中的0值。
在用户没有评级的所有物品中，对每个物品预计一个可能的评级分数。这就是说：我们认为用户可能会对物品的打分（这就是相似度计算的初衷）。
对这些物品的评分从高到低进行排序，返回前N个物品。
'''
# 基于物品相似度的推荐引擎
def standEst(dataMat, user, simMeas, item):
    """standEst(计算某用户未评分物品中，以对该物品和其他物品评分的用户的物品相似度，然后进行综合评分)

    Args:
        dataMat         训练数据集
        user            用户编号
        simMeas         相似度计算方法
        item            未评分的物品编号
    Returns:
        ratSimTotal/simTotal     评分（0～5之间的值）
    """
    # 得到数据集中的物品数目(按物品计算，即案例计算，每一列表示一个物品)
    n = shape(dataMat)[1]
    # 初始化两个评分值
    simTotal = 0.0
    ratSimTotal = 0.0
    # 遍历行中的每个物品（对用户评过分的物品进行遍历，并将它与其他物品进行比较）
    for j in range(n):
        userRating = dataMat[user, j]
        # 如果某个物品的评分值为0，则跳过这个物品
        if userRating == 0:
            continue
        # 寻找两个用户都评级的物品
        # 变量 overLap 给出的是两个物品当中已经被评分（非0元素）的那个元素的索引ID
        # logical_and 计算x1和x2元素的真值。np.logical_and(逻辑与)(两个都大于0，才为真)
        #。A以数组的方式进行求解
        overLap = nonzero(logical_and(dataMat[:, item].A > 0, dataMat[:, j].A > 0))[0]
        # 如果相似度为0，则两着没有任何重合元素，终止本次循环
        if len(overLap) == 0:
            similarity = 0
        # 如果存在重合的物品，则基于这些重合物重新计算相似度。
        else:
            similarity = simMeas(dataMat[overLap, item], dataMat[overLap, j])
        # print('the %d and %d similarity is : %f'(iten,j,similarity))
        # 相似度会不断累加，每次计算时还考虑相似度和当前用户评分的乘积
        # similarity  用户相似度，   userRating 用户评分
        simTotal += similarity        #相似度不断进行累加
        #print(simTotal)
        ratSimTotal += similarity * userRating    #计算相似度和当前用户的乘积,为相似度评分的乘积
    if simTotal == 0:
        return 0
    # 通过除以所有的评分总和，对上述相似度评分的乘积进行归一化，使得最后评分在0~5之间，这些评分用来对预测值进行排序
    else:
        return ratSimTotal/simTotal

'''--------排序获取最后的推荐结果-------------'''
# recommend()函数，就是推荐引擎，它默认调用standEst()函数，产生了最高的N个推荐结果。
# 如果不指定N的大小，则默认值为3。该函数另外的参数还包括相似度计算方法和估计方法
def recommend(dataMat, user, N=3, simMeas=cosSim, estMethod=standEst):
    # 寻找未评级的物品
    # 对给定的用户建立一个未评分的物品列表
    unratedItems = nonzero(dataMat[user, :].A == 0)[1]
    # 如果不存在未评分物品，那么就退出函数
    if len(unratedItems) == 0:
        return 'you rated everything'
    # 物品的编号和评分值
    itemScores = []
    # 在未评分物品上进行循环
    for item in unratedItems:
        estimatedScore = estMethod(dataMat, user, simMeas, item)
        # 寻找前N个未评级物品，调用standEst()来产生该物品的预测得分，该物品的编号和估计值会放在一个元素列表itemScores中
        itemScores.append((item, estimatedScore))
        # 按照估计得分，对该列表按照估计评分进行排序并返回。列表逆排序，第一个值就是最大值
    return sorted(itemScores, key=lambda jj: jj[1], reverse=True)[:N]
  
    
myMat=mat(loadExData())
#对矩阵的值做一些更改
myMat[0,1]=myMat[0,0]=myMat[1,0]=myMat[2,0]=4
myMat[3,3]=2
myMat
#按照默认相似度计算方法进行推荐
recommend(myMat,2)
#结果表明用户2对物品2的预测评分值为2.5，对物品1的预测评分值为2.05

#按照欧氏距离相关系数进行与预测
recommend(myMat,2,simMeas=ecludSim)

#按照皮尔逊距离相关系数进行与预测
recommend(myMat,2,simMeas=pearsSim)

'''----------------------------利用SVD提高推荐的效果-------------------------'''
#使用原因：数据比较稀疏，适合降维处理，提高推荐效果
from numpy import linalg as la
def loadExData2():
    # 书上代码给的示例矩阵
    return[[0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 5],
           [0, 0, 0, 3, 0, 4, 0, 0, 0, 0, 3],
           [0, 0, 0, 0, 4, 0, 0, 1, 0, 4, 0],
           [3, 3, 4, 0, 0, 0, 0, 2, 2, 0, 0],
           [5, 4, 5, 0, 0, 0, 0, 5, 5, 0, 0],
           [0, 0, 0, 0, 5, 0, 1, 0, 0, 5, 0],
           [4, 3, 4, 0, 0, 0, 0, 5, 5, 0, 1],
           [0, 0, 0, 4, 0, 4, 0, 0, 0, 0, 4],
           [0, 0, 0, 2, 0, 2, 5, 0, 0, 1, 2],
           [0, 0, 0, 0, 5, 0, 0, 0, 0, 4, 0],
           [1, 0, 0, 0, 0, 0, 0, 1, 2, 0, 0]]
def loadExData3():
    return[[2,0,0,4,4,0,0,0,0,0,0],
           [0,0,0,0,0,0,0,0,0,0,5],
           [0,0,0,0,0,0,0,1,0,4,0],
           [3,3,4,0,3,0,0,2,2,0,0],
           [5,5,5,0,0,0,0,0,0,0,0],
           [0,0,0,0,0,0,5,0,0,5,0],
           [4,0,4,0,0,0,0,0,0,0,5],
           [0,0,0,0,0,4,0,0,0,0,4],
           [0,0,0,0,0,0,5,0,0,5,0],
           [0,0,0,3,0,0,0,0,4,5,0],
           [1,1,2,1,1,2,1,0,4,5,0]]
U,Sigma,VT=la.svd(mat(loadExData2()))
Sigma
myMat=mat(loadExData2())
#查看有多少奇异值能达到总能量的90%
#分析数据，多少数据可以满足指定信息量的要求
def analyse_data(Sigma, loopNum=20):
    """analyse_data(分析 Sigma 的长度取值)
    Args:
        Sigma         Sigma的值
        loopNum       循环次数
    """
    # 总方差的集合（总能量值）
    Sig2 = Sigma**2
    SigmaSum = sum(Sig2)
    for i in range(loopNum):
        SigmaI = sum(Sig2[:i+1])
        '''
        根据自己的业务情况，就行处理，设置对应的 Singma 次数
        通常保留矩阵 80% ～ 90% 的能量，就可以得到重要的特征并取出噪声。
        '''
        print('主成分：%s, 方差占比：%s%%' % (format(i+1, '2.0f'), format(SigmaI/SigmaSum*100, '4.2f')))
analyse_data(Sigma, loopNum=20)

'''-----------------------基于SVD的评分估计-------------------------------'''
# 在recommend() 中，这个函数用于替换对standEst()的调用，该函数对给定用户给定物品构建了一个评分估计值

U,Sigma,VT=la.svd(mat(loadExData3()))
Sigma

def svdEst(dataMat, user, simMeas, item):
    """svdEst( )
    Args:
       dataMat         训练数据集
        user            用户编号
        simMeas         相似度计算方法
       item            未评分的物品编号
    Returns:
        ratSimTotal/simTotal     评分（0～5之间的值）
    """
    # 物品数目
    n = shape(dataMat)[1]
    # 对数据集进行SVD分解
    simTotal = 0.0
    ratSimTotal = 0.0
    # 奇异值分解
    # 在SVD分解之后，我们只利用包含了90%能量值的奇异值，这些奇异值会以NumPy数组的形式得以保存
    U, Sigma, VT = la.svd(dataMat)
    # # 分析 Sigma 的长度取值
    # analyse_data(Sigma, 20)
    # 如果要进行矩阵运算，就必须要用这些奇异值构建出一个对角矩阵
    Sig4 = mat(eye(4) * Sigma[: 4])     #获取前四个奇异值
    # 利用U矩阵将物品转换到低维空间中，构建转换后的物品(物品+4个主要的特征（物品数不变，表述物品的特征减小到4）)
    xformedItems = dataMat.T * U[:, :4] * Sig4.I 
    print('dataMat', shape(dataMat))
    print('U[:, :4]', shape(U[:, :4]))
    print('Sig4.I', shape(Sig4.I))
    print('VT[:4, :]', shape(VT[:4, :]))
    print('xformedItems', shape(xformedItems))
    # 对于给定的用户，for循环在用户对应行的元素上进行遍历
    # 这和standEst()函数中的for循环的目的一样，只不过这里的相似度计算时在低维空间下进行的。
    for j in range(n):
        userRating = dataMat[user, j]
        if userRating == 0 or j == item:    #对于所有标记过的标记值非0商品，与未标记过的商品，计算相关系数
            continue
        # 相似度的计算方法也会作为一个参数传递给该函数
        similarity = simMeas(xformedItems[item, :].T, xformedItems[j, :].T)
        # for 循环中加入了一条print语句，以便了解相似度计算的进展情况。如果觉得累赘，可以去掉
        print('the %d and %d similarity is: %f' % (item, j, similarity))
        # 对相似度不断累加求和
        simTotal += similarity
        ratSimTotal += similarity * userRating
    if simTotal == 0:
        return 0
    else:
        # 计算估计评分，评分为表明了标记商品与未标记商品的相似度，越大评分越高，越推荐
        return ratSimTotal/simTotal



#分析数据，多少数据可以满足指定信息量的要求
def analyse_data(Sigma, loopNum=20):
    """analyse_data(分析 Sigma 的长度取值)
    Args:
        Sigma         Sigma的值
        loopNum       循环次数
    """
    # 总方差的集合（总能量值）
    Sig2 = Sigma**2             #求奇异值的平方
    SigmaSum = sum(Sig2)         #计算数据总能量
    for i in range(loopNum):      
        SigmaI = sum(Sig2[:i+1])    #计算前i个奇异值的累计能量
        '''
        根据自己的业务情况，就行处理，设置对应的 Singma 次数
        通常保留矩阵 80% ～ 90% 的能量，就可以得到重要的特征并取出噪声。
        '''
        print('主成分：%s, 方差占比：%s%%' % (format(i+1, '2.0f'), format(SigmaI/SigmaSum*100, '4.2f')))

analyse_data(Sigma, loopNum=20)


myMat=mat(loadExData2())
#应用推荐函数为用户1推荐商品
recommend(myMat,1,estMethod=svdEst,simMeas=pearsSim)
recommend(myMat,1,estMethod=standEst)
'''
SVD推荐引擎的问题：
1.在大数据集上运行很慢，每天运行一次，或者频率更低，并且还要离线运行
2.实际系统中0的数目很多，为节省内存，可否只存储非0元素
3.每次推荐一个物品的得分，就要计算多个物品的相似度得分，这些得分记录的是物品间的相似度，这些得分可以被另一个用户重复使用
4.现实中普遍的解决办法就是：离线计算并保存相似度得分
5.推荐引擎的另一个问题就是冷启动问题，（如何在缺乏数据时给出好的推荐）
6.解决方法：基于内容的商品推荐，
'''
'''------------------基于SVD的图像压缩-----------------------------------'''
'''
数据：手写数字图像，32*32=1024像素
对数据进行压缩，使用更少的像素来表示此图片
方法：SVD
'''
# 图像压缩函数

# 加载并转换数据

def imgLoadData(filename):
    myl = []
    # 打开文本文件，并从文件以数组方式读入字符
    for line in open(filename).readlines():
        newRow = []
        for i in range(32):
            newRow.append(int(line[i]))
        myl.append(newRow)
    # 矩阵调入后，就可以在屏幕上输出该矩阵
    myMat = mat(myl)
    return myMat


# 打印矩阵

def printMat(inMat, thresh=0.8):
# 由于矩阵包含了浮点数，因此定义浅色和深色，遍历所有矩阵元素，当元素大于阀值时打印1，否则打印0
    for i in range(32):
        for k in range(32):
            if float(inMat[i, k]) > thresh:
                print(1, end=' ')
            else:
                print(0, end=' ')
        print('')


# 实现图像压缩，允许基于任意给定的奇异值数目来重构图像
def imgCompress(numSV=3, thresh=0.8):
    """imgCompress( )
    Args:
        numSV       Sigma长度   
        thresh      判断的阈值
    """
    # 构建一个列表
    my1=[]
    myMat = imgLoadData(r'E:\python_study\Python_work\data\14\0_5.txt')
    print("****original matrix****")
    # 对原始图像进行SVD分解并重构图像e
    printMat(myMat, thresh)
    # 通过Sigma 重新构成SigRecom来实现
    # Sigma是一个对角矩阵，因此需要建立一个全0矩阵，然后将前面的那些奇异值填充到对角线上。
    U, Sigma, VT = la.svd(myMat)
    #SigRecon = mat(zeros((numSV, numSV)))
    #for k in range(numSV):
     #   SigRecon[k, k] = Sigma[k]
    # 分析插入的 Sigma 长度
    analyse_data(Sigma, 20)
    SigRecon = mat(eye(numSV) * Sigma[: numSV])
    reconMat = U[:, :numSV] * SigRecon * VT[:numSV, :]  
    #根据奇异值获得近似矩阵（即压缩后的矩阵）
    print("****reconstructed matrix using %d singular values *****" % numSV)
    printMat(reconMat, thresh)

#用两个奇异值压缩，U和VT均为32*2的矩阵，压缩后的像素为64+64+2=130个数目 


if __name__ == "__main__":

    # # 对矩阵进行SVD分解(用python实现SVD)
    # Data = loadExData()
    # print('Data:', Data)
    # U, Sigma, VT = linalg.svd(Data)
    # # 打印Sigma的结果，因为前3个数值比其他的值大了很多，为9.72140007e+00，5.29397912e+00，6.84226362e-01
    # # 后两个值比较小，每台机器输出结果可能有不同可以将这两个值去掉
    # print('U:', U)
    # print('Sigma', Sigma)
    # print('VT:', VT)
    # print('VT:', VT.T)
    # # 重构一个3x3的矩阵Sig3
    # Sig3 = mat([[Sigma[0], 0, 0], [0, Sigma[1], 0], [0, 0, Sigma[2]]])
    # print(U[:, :3] * Sig3 * VT[:3, :])
    """    
    # 计算欧氏距离
    myMat = mat(loadExData())
    #print(myMat)
    print(ecludSim(myMat[:, 0], myMat[:, 4]))
    print(ecludSim(myMat[:, 0], myMat[:, 0]))
    # 计算余弦相似度
    print(cosSim(myMat[:, 0], myMat[:, 4]))
    print(cosSim(myMat[:, 0], myMat[:, 0]))
    # 计算皮尔逊相关系数
    print(pearsSim(myMat[:, 0], myMat[:, 4]))
    print(pearsSim(myMat[:, 0], myMat[:, 0]))
    """
    # 计算相似度的方法
    myMat = mat(loadExData3())
    # 计算相似度的第一种方式
    print(recommend(myMat, 1, estMethod=svdEst))
    # 计算相似度的第二种方式
    print(recommend(myMat, 1, estMethod=svdEst, simMeas=pearsSim))
    # 默认推荐（菜馆菜肴推荐示例）
    print(recommend(myMat, 2))
    """
    # 利用SVD提高推荐效果
    U, Sigma, VT = la.svd(mat(loadExData2()))
    print(Sigma)                # 计算矩阵的SVD来了解其需要多少维的特征
    Sig2 = Sigma**2             # 计算需要多少个奇异值能达到总能量的90%
    print(sum(Sig2))            # 计算总能量
    print(sum(Sig2) * 0.9)       # 计算总能量的90%
    print(sum(Sig2[: 2]))        # 计算前两个元素所包含的能量
    print(sum(Sig2[: 3]))        # 两个元素的能量值小于总能量的90%，于是计算前三个元素所包含的能量
    # 该值高于总能量的90%，这就可以了
    """
    # 压缩图片
imgCompress(2)
