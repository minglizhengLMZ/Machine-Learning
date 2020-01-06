# -*- coding: utf-8 -*-
"""
Created on Fri Nov  1 11:46:56 2019

@author: Administrator
"""

'''
比较不同分类器的原则
kNN：计算样本间的距离，判为最近类所属类别，多少训练样本，则计算多少次距离（计算量大）
决策树：分别沿x轴和y轴划分数据 (分类效果不一定好)
贝叶斯：计算样本点属于每个类的概率，到哪个类的概率大则判它属于哪类

朴素贝叶斯：使用概率论进行分类的方法  文档分类的常用算法
优点：数据较少的情况下有效，可处理多类别问题
缺点：对于输入数据的真被方式比较敏感
使用数据类型：标称型数据

贝叶斯决策的判定准则:
    p(c1|x,y)>p(c2|x,y)  判为第一类
    p(c1|x,y)<p(c2|x,y)  判为第二类
朴素贝叶斯的一般过程
（1）搜集数据
（2）准备数据:需要数值型或者布尔型数据
（3）分析数据：有大量特征时绘制特征作用不大，使用直方图效果较好
（4）训练算法：计算不同独立特征的条件概率
（5）测试算法：计算错误率
（6）使用算法：常见的朴素贝叶斯应用是文档分类，可在任意分类的文档中使用
朴素贝叶斯的假设
1.特征之间相互独立，1000个特征，需要1000*N个样本
2.特征同等重要 
尽管两个假设都存在问题，在实际中应用效果却很好

'''

#1.从文本中构造词向量
'''----------------词表到向量的准换函数-------------'''
def loadDataSet():
    postingList=[['my','dog','has','flea',
                  'problems','help','please'],
                 ['maybe','not','take','him',
                  'to','dog','park','stupid'],
                 ['my','dalmation','is','so','cute',
                   'I','love','him'],
                 ['stop','posting','stupid','worthless','garbage'],
                 ['mr','licks','ate','my','steak','how',
                  'to','stop','him'],
                 ['quit','buying','worthless','dog','food','stupid']]
    classVec=[0,1,0,1,0,1]   #1表示侮辱形文字，0表示正常言论
    return postingList,classVec
#该函数返回的第一个变量是进行词条切割后的文档集合
#第二个变量是一个类别标签的集合
'''---------------创建不包含重复数的集合(词汇表)------------------'''
def createVocaList(dataSet):
    vocabSet = set([])
    for document in dataSet:
        vocabSet = vocabSet | set(document) 
 #|或，用于求两个集合的并集，在原来基础上不断加入新词
    return list(vocabSet)
'''创建包含在文档中的不重复词的集合（set函数自动去重）、
创建空的set集合
将文档返回的新词添加到set集合中
'''
'''---------------检查词汇表中的单词是否在文档中出现------------'''
#词集模型，每个词的出现与否作为特征
def setOfWords2Vec(vocabList,inputSet):  #输入参数词汇表，文档
    returnVec = [0]*len(vocabList)  #创建一个所含元素都为0的向量,长度与词汇表相同
    for word in inputSet:  #历遍文档的单词         
        if word in vocabList:  #如果文档的单词在词汇表中出现
            returnVec[vocabList.index(word)]=1  
  #相应return相应单词的索引对应的值变为1，（词汇表中未出现的单词默认为0）
        else:    
            print("the word:'%s' is not in my Vocabulary!"%word)  #
    return returnVec   
    
'''--------------测试函数---------------'''
import bayes
listOPosts,listClasses = bayes.loadDataSet()  #导入文档，并转化为向量形式
myVocabList = bayes.createVocaList(listOPosts)  #创建词汇表
myVocabList  #词汇表
len(myVocabList)
bayes.setOfWords2Vec(myVocabList,listOPosts[0])   
bayes.setOfWords2Vec(myVocabList,listOPosts[3])   



#数据分为两个两个等级
#1.文档，文档类别，向量
#2.词汇，词汇数，侮辱性词汇，正常词汇，单词'''

'''--------------训练算法:从词向量计算概率---------------'''
#伪代码见书和笔记
'''计算每个类别的文档数
对每片训练文档:
    对每个类别:
        如果词条出现在文档中:
            增加该词条的计数值（计算改词出现次数）
            增加所有词条的计数值（计算该类别的词汇数）
    对每个类别:
        对每个词条:
            条件概率=该词条数目/该类别总数
    返回每个类别的条件概率'''

'''-------------计算条件概率------------------'''
from numpy import *   #一般将掉包工作放在文档的最前端
def trainNB0(trainMatrix,trainCategory):  #训练算法，参数为文档矩阵(0-1变量)和每个文档的类别标签
    numTrainDocs = len(trainMatrix)      #计算文档个数即向量数，作为分母
    numWords = len(trainMatrix[0])       ##单词数量
    pAbusive = sum(trainCategory)/float(numTrainDocs) ##统计侮辱性文档总个数，然后除以总文档个数
    p0Num = ones(numWords);p1Num = ones(numWords)  #分别创建与单词数相等的数组
    p0Denom = 2.0;p1Denom = 2.0         #设定初始值
    #向量的条件概率为各分量的条件概率之积，避免某一概率为0，也避免分母为0，如此设定初始值
    for i in range(numTrainDocs):       #历遍训练集中的所有文档
        if trainCategory[i]==1:        #如果是侮辱性文档
            p1Num +=trainMatrix[i]          ##把属于同一类的文本向量相加，实质是统计某个词条在该文本类中出现的频率
            p1Denom += sum(trainMatrix[i]) ##去重
        else:  
            p0Num +=trainMatrix[i]   #如果是类别0 ，则归入0的数组，累计计算该数组词目数
            p0Denom += sum(trainMatrix[i])  
    p1Vect = log(p1Num/p1Denom )   #分别统计统计词典中所有词条在侮辱性文档中出现的概率
    p0Vect =log(p0Num/p0Denom)  
#概率太小容易导致下溢，取自然对数可以避免下溢，或者浮点数四舍五入导致的错误    
    return p0Vect,p1Vect,pAbusive

'''测试代码'''
import bayes
listOPosts,listClasses = bayes.loadDataSet()  #得到文档，及其类别以向量形式展示
myVocabList = bayes.createVocaList(listOPosts)  #创建词汇表
#创建包含所有词的列表myVocabList

trainMat=[]
for postinDoc in listOPosts:
    trainMat.append(bayes.setOfWords2Vec(myVocabList,postinDoc)) #传入标签和所有的文档，构建0-1数据列表代表文档词汇
    
#给出侮辱性文档的词汇，以及两个类别的概率向量
p0v,p1v,pAb=bayes.trainNB0(trainMat,listClasses)  #统计各侮辱性词汇及正常词汇所占比例，及侮辱性文档占所有文档的比例
p0v                                              
p1v
pAb

'''---------------------朴素贝叶斯分类函数----------------'''
def classifyNB(vec2Classify,p0Vec,p1Vec,pClass1): 
 #参数为要分类的向量，经过（setOfWords2Vec变换为0-1元素的向量），其元素与词汇表顺序相对应
 #以及trainNB0()计算得到的三个概率，各类别中各词汇出现概率，以及侮辱性文档所占比例
    p1 = sum(vec2Classify*p1Vec)+log(pClass1)   
#计算两个向量的对应元素相乘，加上log(a*b)=log(a)+log(b),两个条件概率分母相同，只比较分子大小即可
    p0 = sum(vec2Classify*p0Vec)+log(1-pClass1)
    if p1>p0:
        return 1
    else:
        return 2
    
def testingNB():
    listOPosts,listClasses = loadDataSet()
    myVocabList = createVocaList(listOPosts)
    trainMat=[]
    for postinDoc in listOPosts:
        trainMat.append(setOfWords2Vec(myVocabList,postinDoc))
    p0v,p1v,pAb= trainNB0(array(trainMat),array(listClasses))
    testEntry = ['love','my','dalmation']
    thisDoc = array(setOfWords2Vec(myVocabList,testEntry))
    print (testEntry,'classified as:',classifyNB(thisDoc,p0v,p1v,pAb))
    testEntry = ['stupid','garbage']
    thisDoc=array(setOfWords2Vec(myVocabList,testEntry))
    print(testEntry,'classified as:',classifyNB(thisDoc,p0v,p1v,pAb))
        
import bayes
bayes.testingNB()        
 

'''---------------------上述贝叶斯的改进(对setOfWord2Vec()的改进)----------------'''       
#准备数据;文档词袋模型,包含词在文本中出现次数
def bagOfWords2VecMN(vocabList,inputSet):
    returnVec = [0]*len[vocabList]
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] +=1
            return returnVec
        
        
'''----------------------使用朴素贝叶斯过滤垃圾文件---------------
（1）搜集数据：提供文本文件
（2）准备数据:将数据解析成词条向量
（3）分析数据：检查词条确保解析的重要性 
（4）训练算法：使用我们之前建立的trainNB0()函数
（5）测试算法：使用classifyNB(),并构建一个新的测试函数来计算文档集的错误率
（6）使用算法：构建一个完整的程序对一组文档进行分类，将错分文档输出到屏幕上
以下是邮件储存路径格式，含“spms”的文件为垃圾文件。'''
#准备数据：切分文本（创建词表）
#如何创建词向量，并基于这些词向量进行朴素贝叶斯分类过程

'''------------#4.6.1利用string.split()分割文本--------'''
mySent = 'This book is the best book on python or M.L. I have ever laid eyes upon.'
mySent.split()   #split()将一个字符串切割成词条向量
#上例将标点符号当成了词的一部分

#正则表达式切割句子，分隔符是除单词、数字外的任意字符串
import re    #正则化，在文本处理中经常用到
regex=re.compile('(\w*)') #同compile编译regex，实现其可重复利用   将源代码编译成可由exec()或eval()执行的代码对象。
listOfTokens = regex.split(mySent)
listOfTokens

#去掉词表的空字符串
[tok for tok in listOfTokens if len(tok)>0]
#将字符串全部转换成小写或者大写
[tok.lower() for tok in listOfTokens if len(tok)>0]
[tok.upper() for tok in listOfTokens if len(tok)>0]

#读取数据,6.text里存储了某公司不再支持的文件
emilText=open(r'E:\Anaconda3\Python_work\data\6.txt').read()
listOfTokens=re.split('(\w*)',emilText)
listOfTokens
#将文本解析器集成一个完整贝叶斯分类器
#文本分割
def textParse(bigString):
    import re
    listOfTokens = re.split('(\w*)',bigString)
    return [tok.lower() for tok in listOfTokens if len(tok)>2]

def spamTest():
    docList=[];classList=[];fullText =[]
    for i in range(1,26):
       #导入并解析数据
        wordList = textParse(open(r'E:\Anaconda3\Python_work\data\spam\%d.txt' %i).read())
        docList.append(wordList)  #append操作对象可任意
        fullText.extend(wordList)  #extend对象只能为列表
        # extend（) 函数用于在列表末尾一次性追加另一个序列中的多个值（用新列表扩展原来的列表）
        classList.append(1) #分类类列表用0 填充，实现与i长度相同的列表
        wordList = textParse(open(r'E:\Anaconda3\Python_work\data\ham\%d.txt' %i).read())
        docList.append(wordList)  #append操作对象可任意
        fullText.extend(wordList)  #extend对象只能为列表
        # extend（) 函数用于在列表末尾一次性追加另一个序列中的多个值（用新列表扩展原来的列表）
        classList.append(0) #分类类列表用0 填充，实现与i长度相同的列表        
    vocabList = createVocaList(docList)  #创建词汇表
    trainingSet = list(range(50));testSet=[]   #设定训练样本数，检验集
    print(docList)
    print(classList)
   #随机创建训练集
    for i in range(10):
        randIndex = int(random.uniform(0,len(trainingSet)))  #随机构造训练集
        testSet.append(trainingSet[randIndex])  
        del(trainingSet[randIndex])   #将已抽样个体从总体删除，不重复抽样
    
    trainMat=[];trainClasses =[]   
    for docIndex in trainingSet:
        #检查文档单词是否在词汇表出现，0-1
        trainMat.append(setOfWords2Vec(vocabList,docList[docIndex])) #逐个检验
        trainClasses.append(classList[docIndex])
   #计算条件概率
    p0V,p1V,pSpam = trainNB0(array(trainMat),array(trainClasses)) #训练算法实现分类
    #分析错误率 
    errorCount = 0
    for docIndex in testSet:
        wordVector = setOfWords2Vec(vocabList,docList[docIndex])
        if classifyNB(array(wordVector),p0V,p1V,pSpam) !=classList[docIndex]:
            errorCount +=1
    print('the erro rate is:',float(errorCount)/len(testSet))

import bayes  
bayes.spamTest()
bayes.spamTest()   
  



'''------使用朴素贝叶斯分类器从个人广告中获取区域倾向-------'''
#比较两个地方的人，在征婚广告用词上是否不同
#如果不同，找到各地区的常用词，了解各地区人所关心的内容


          
        
        
        
        
        
        
        
        

