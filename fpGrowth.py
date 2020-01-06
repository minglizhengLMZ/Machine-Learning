# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 10:24:58 2019

@author: lenovo
"""

'''
在 第11章 时我们已经介绍了用 Apriori 算法发现 频繁项集 与 关联规则。
本章将继续关注发现 频繁项集 这一任务，并使用 FP-growth 算法更有效的挖掘 频繁项集。

FP-growth 算法简介:
一种非常好的发现频繁项集算法。
基于Apriori算法构建,但是数据结构不同，使用叫做 FP树 的数据结构结构来存储集合。下面我们会介绍这种数据结构。

FP-growth 算法步骤:
基于数据构建FP树 
从FP树种挖掘频繁项集 

FP-growth算法
优点：一般要快与Apriori
缺点：实现比较困难，在某些数据集上性能会下降
使用数据类型：标称型数据

FP-growth的一般流程：
（1）搜集数据：使用任意方法
（2）准备数据:由于存储的是集合，需要离散数据，如果是连续数据，需要将连续数据离散化，pandas的数据离散化方法
（3）分析数据：使用任意方法
（4）训练算法：构建一个FP树，并对树进行挖掘
（5）测试算法：没有测试过程
（6）使用算法：用于识别经常出现的元素，从而用于制定决策，推荐元素，或进行预测等应用中，如：搜索引擎公司

FP-growth算法的工作流程：
1.构建FP树，利用它来挖掘频繁项集
对原始数据集扫描两边
（1）：对所有元素项的出现次数进行计数（Apriori原理），获得频繁项集
（2）：对频繁项集进行第二遍扫描
'''
#构建一个容器来保存FP树，此处构建一个类来保存树的每个节点
'''
类的部分知识总结：
实例的数据成员一般是指在构造函数__init__()中定义的，定义和使用时必须以self作为前缀；
属于类的数据成员是在类中所有方法之外定义的。

'''
class treeNode:

    def __init__(self, nameValue, numOccur, parentNode):   #类的数据成员，定义实例属性
        self.name = nameValue          #存放节点名的变量
        self.count = numOccur          #计数值
        self.nodeLink = None          #链接link，链接相似元素
        # needs to be updated
        self.parent = parentNode      #父变量来指定当前节点的父节点
        self.children = {}               #空字典用于存放父节点的子节点
    def inc(self, numOccur):            #定义类的成员方法
        """inc(对count变量增加给定值)
        """
        self.count += numOccur
    def disp(self, ind=1):
        """disp(用于将树以文本形式显示)，此方法主要对调试过程非常有用
        """
        print('  '*ind, self.name, ' ', self.count)
        for child in self.children.values():
            child.disp(ind+1)
#创建树的一个单节点
rootNode=treeNode('pyramid',9,None)      #实例化
#为其增加子节点
rootNode.children['eye']=treeNode('eye',13,None)  #调用类的数据成员
#显示子节点
rootNode.disp()     #调用成员方法，展示树的函数
#再增加一个子节点
rootNode.children['phoenix']=treeNode('phoenix',3,None)  #调用类的数据成员
#调用展示树的成员方法
rootNode.disp()

'''
1.对事务记录进行过滤排序：

遍历所有的数据集合，计算所有项的支持度。
丢弃非频繁的项。
基于 支持度 降序排序所有的项。 
所有数据集合按照得到的顺序重新整理。
重新整理完成后，丢弃每个集合末尾非频繁的项。 

2.构建FP树
从空集开始向其不断添加频繁项集
排序过滤后的事务依次添加到书中
（树中有该元素增加值，书中无该元素向树添加分支）

3.使用字段作为数据结构保存头指针表

'''
'''-----------------------------FP树的构建函数-------------------------------'''
def createTree(dataSet, minSup=1):
    """createTree(生成FP-tree)
    Args:
        dataSet  dist{行：出现次数}的样本数据
        minSup   最小的支持度
    Returns:
        retTree  FP-tree
        headerTable 满足minSup {所有的元素+(value, treeNode)}
    """
    # 支持度>=minSup的dist{所有元素：出现的次数}
    headerTable = {}    #构建头指针表
    # 循环 dist{行：出现次数}的样本数据
    for trans in dataSet:
        # 对所有的行进行循环，得到行里面的所有元素
        # 统计每一行中，每个元素出现的总次数
        for item in trans:
            # 例如： {'ababa': 3}  count(a)=3+3+3=9   count(b)=3+3=6   ???
            headerTable[item] = headerTable.get(item, 0) + dataSet[trans]  
            #.get(item,0)函数的作用，返回字典item元素对应的值，若无则进行初始化，并增加dataSet的trans对应的值
   
    # 删除 headerTable中，元素次数<最小支持度的元素
    for k in list(headerTable.keys()):
        if headerTable[k] < minSup:
            del(headerTable[k])

    # 满足minSup: set(各元素集合),对key去重
    freqItemSet = set(headerTable.keys())
    # 如果没有元素满足要求，直接返回None，退出循环
    if len(freqItemSet) == 0:
        return None, None 
    #对字典的value进行格式化，转化成含两个元素的列表形式，目的：存储元素的次数，及节点
    for k in headerTable:
        # 格式化： dist{元素key: [元素次数, None]}
        headerTable[k] = [headerTable[k], None]    

    # create tree
    retTree = treeNode('Null Set', 1, None)       #创建只包含空集合的根节点
    # 循环 dist{行：出现次数}的样本数据
    for tranSet, count in dataSet.items():
        # print('tranSet, count=', tranSet, count)
        # localD = dist{元素key: 元素总出现次数}
        localD = {}
        for item in tranSet:
            # 判断是否在满足minSup的集合中
            if item in freqItemSet:
                # print('headerTable[item][0]=', headerTable[item][0], headerTable[item])
                localD[item] = headerTable[item][0]   #将item该键对应的一个也value值传到localD中
        # print('localD=', localD)
        if len(localD) > 0:
            # p=key,value; 所以是通过value值的大小，进行从大到小进行排序
            # orderedItems 表示取出元组的key值，也就是字母本身，但是字母本身是大到小的顺序
            orderedItems = [v[0] for v in sorted(localD.items(), key=lambda p: p[1], reverse=True)]
            # print('orderedItems=', orderedItems, 'headerTable', headerTable, '\n\n\n')
            # 填充树，通过有序的orderedItems的第一位，进行顺序填充 第一层的子节点。
            updateTree(orderedItems, retTree, headerTable, count)
            #orderedItems为满足阙值条件的，包含key的元组，retTree空的树对象，
            #headerTable：满足阙值条件的元素，统计次数，及对应的树节点
    return retTree, headerTable


def updateTree(items, inTree, headerTable, count):

    """updateTree(更新FP-tree，第二次遍历)
    # 针对每一行的数据

    # 最大的key,  添加
    Args:
        items       满足minSup 排序后的元素key的数组（大到小的排序）
        inTree      空的Tree对象
        headerTable 满足minSup {所有的元素+(value, treeNode)}
        count       原数据集中每一组Kay出现的次数
    """
    # 取出 元素 出现次数最高的
    # 如果该元素在 inTree.children 这个字典中，就进行累加
    # 如果该元素不存在 就 inTree.children 字典中新增key，value为初始化的 treeNode 对象
    if items[0] in inTree.children:      #判断一个事务是否作为子节点存在
        # 更新 最大元素，对应的 treeNode 对象的count进行叠加
        inTree.children[items[0]].inc(count)
    else:
        # 如果不存在子节点，我们为该inTree添加子节点，通过treeNode创建节点，并将其作为子节点添加到树中
        inTree.children[items[0]] = treeNode(items[0], count, inTree)
        # 如果满足minSup的dist字典的value值第二位为null， 我们就设置该元素为 本节点对应的tree节点
        # 如果元素第二位不为null，我们就更新header节点
        if headerTable[items[0]][1] == None:
            # headerTable只记录第一次节点出现的位置
            headerTable[items[0]][1] = inTree.children[items[0]]
        else:
            # 本质上是修改headerTable的key对应的Tree，的nodeLink值，头指针也要更新以指向新的节点
            updateHeader(headerTable[items[0]][1], inTree.children[items[0]])
    if len(items) > 1:
        # 递归的调用，在items[0]的基础上，添加item0[1]做子节点， count只要循环的进行累计加和而已，统计出节点的最后的统计值。
        updateTree(items[1::], inTree.children[items[0]], headerTable, count)



def updateHeader(nodeToTest, targetNode):

    """updateHeader(更新头指针，建立相同元素之间的关系，
    例如： 左边的r指向右边的r值，就是后出现的相同元素 指向 已经出现的元素)
    从头指针的nodeLink开始，一直沿着nodeLink直到到达链表末尾。这就是链表。
    性能：如果链表很长可能会遇到迭代调用的次数限制。
    Args:
        nodeToTest  满足minSup {所有的元素+(value, treeNode)}
        targetNode  Tree对象的子节点
    """
    # 建立相同元素之间的关系，例如： 左边的r指向右边的r值
    while (nodeToTest.nodeLink != None):
        nodeToTest = nodeToTest.nodeLink
    nodeToTest.nodeLink = targetNode
# 此版本不适用递归
#创建简单的数据集
def loadSimpDat():
    simpDat=[['r','z','h','j','p'],
             ['z','y','x','w','v','u','t','s'],
             ['z'],
             ['r','x','n','o','s'],
             ['y','r','x','z','q','t','p'],
             ['y','z','x','e','q','s','t','m']]
    return simpDat
#数据包装器
def createInitSet(dataSet):
    retDict = {}
    for trans in dataSet:
        if frozenset(trans) not in retDict :
            retDict[frozenset(trans)] = 1
        else:
            retDict[frozenset(trans)] += 1
    return retDict
#导入数据实力
simpDat=loadSimpDat()
simpDat
#对上面数据进行格式化处理
initSet=createInitSet(simpDat)
initSet
#创造FP树
myFPtree,myHeaderTab=createTree(initSet,3)
#以文本形式展示树，每个缩进表示所处树的深度
myFPtree.disp()
'''
从一颗FP树中挖掘频繁项集
（1）从FP树中获得条件模式基
（2）利用条件模式基，构建条件FP树
（3）迭代重复步骤1，步骤2，直到树包含一个元素项为止
'''
#发现给定元素项结尾的所有路径的函数
'''
利用先前所创建的头指针表，包含相同类型元素的起始指针;
一旦到达每个元素项，就可以上溯这个树，直到根节点为止
'''

#迭代上溯整棵树，，找到该元素结尾的所有路径
def ascendTree(leafNode, prefixPath):
    """ascendTree(如果存在父节点，就记录当前节点的name值)
    Args:
        leafNode   查询的节点对于的nodeTree
        prefixPath 要查询的节点值
    """
    if leafNode.parent is not None:
        prefixPath.append(leafNode.name)
        ascendTree(leafNode.parent, prefixPath)  


#历遍列表直到结尾，每遇到一个元素都上溯FP树，搜集所遇到元素项的名称，该列表返回之后添加到条件模式基字典conddPats中
def findPrefixPath(basePat, treeNode):
    """findPrefixPath 基础数据集
    Args:
        basePat  要查询的节点值
        treeNode 查询的节点所在的当前nodeTree
    Returns:
        condPats 对非basePat的倒叙值作为key,赋值为count数
    """
    condPats = {}
    # 对 treeNode的link进行循环
    while treeNode != None:
        prefixPath = []
        # 寻找改节点的父节点，相当于找到了该节点的频繁项集
        ascendTree(treeNode, prefixPath)
        # 避免 单独`Z`一个元素，添加了空节点
        if len(prefixPath) > 1:
            # 对非basePat的倒叙值作为key,赋值为count数
            # prefixPath[1:] 变frozenset后，字母就变无序了
            # condPats[frozenset(prefixPath)] = treeNode.count
            condPats[frozenset(prefixPath[1:])] = treeNode.count
        # 递归，寻找改节点的下一个 相同值的链接节点
        treeNode = treeNode.nodeLink
        #print(treeNode)
    return condPats

findPrefixPath('x',myHeaderTab['x'][1])
findPrefixPath('z',myHeaderTab['z'][1])
findPrefixPath('r',myHeaderTab['r'][1])
#和课本结果不同怎么回事？？

'''--------_递归查找频繁项集的mineTree函数-----------------------------'''
def mineTree(inTree, headerTable, minSup, preFix, freqItemList):
    """mineTree(创建条件FP树)
    Args:
        inTree       myFPtree
        headerTable  满足minSup {所有的元素+(value, treeNode)}
        minSup       最小支持项集
        preFix       preFix为newFreqSet上一次的存储记录，一旦没有myHead，就不会更新
        freqItemList 用来存储频繁子项的列表

    """
    # 通过value进行从小到大的排序， 得到频繁项集的key
    # 最小支持项集的key的list集合
    bigL = [v[0] for v in sorted(headerTable.items(), key=lambda p: p[1])]
    print('-----', sorted(headerTable.items(), key=lambda p: p[1]))
    print('bigL=', bigL)
    # 循环遍历 最频繁项集的key，从小到大的递归寻找对应的频繁项集
    for basePat in bigL:
        # preFix为newFreqSet上一次的存储记录，一旦没有myHead，就不会更新
        newFreqSet = preFix.copy()
        newFreqSet.add(basePat)
        print('newFreqSet=', newFreqSet, preFix)
        freqItemList.append(newFreqSet)
        print('freqItemList=', freqItemList)
        condPattBases = findPrefixPath(basePat, headerTable[basePat][1])
        print('condPattBases=', basePat, condPattBases)
         # 构建FP-tree
        myCondTree, myHead = createTree(condPattBases, minSup)
        print('myHead=', myHead)
        # 挖掘条件 FP-tree, 如果myHead不为空，表示满足minSup {所有的元素+(value, treeNode)}
        if myHead != None:
            myCondTree.disp(1)
            print('\n\n\n')
            # 递归 myHead 找出频繁项集
            mineTree(myCondTree, myHead, minSup, newFreqSet, freqItemList)
        print('\n\n\n')

'''

import twitter
from time import sleep
import re



def getLotsOfTweets(searchStr):
    
    """
    获取 100个搜索结果页    
    """
    CONSUMER_KEY = 'get when you create an app'
    CONSUMER_SECRET = 'get when you create an app'
    ACCESS_TOKEN_KEY = 'get from Oauth,specific to a user'
    ACCESS_TOKEN_SECRET = 'get from Oauth,specific to a user'
    api = twitter.Api(consumer_key=CONSUMER_KEY, consumer_secret=CONSUMER_SECRET, access_token_key=ACCESS_TOKEN_KEY, access_token_secret=ACCESS_TOKEN_SECRET)
     # you can get 1500 results 15 pages * 100 per page
    resultsPages = []
    for i in range(1, 15):        
        print("fetching page %d" % i)
        searchResults = api.GetSearch(searchStr, per_page=100, page=i)
        resultsPages.append(searchResults)
        sleep(6)
    return resultsPages

def textParse(bigString):
    """
    解析页面内容
    """
    urlsRemoved = re.sub('(http:[/][/]|www.)([a-z]|[A-Z]|[0-9]|[/.]|[~])*', '', bigString)    
    listOfTokens = re.split(r'\W*', urlsRemoved)
    return [tok.lower() for tok in listOfTokens if len(tok) > 2]


def mineTweets(tweetArr, minSup=5):
    """
    获取频繁项集
    """
    parsedList = []
    for i in range(14):
        for j in range(100):
            parsedList.append(textParse(tweetArr[i][j].text))
    initSet = createInitSet(parsedList)
    myFPtree, myHeaderTab = createTree(initSet, minSup)
    myFreqList = []
    mineTree(myFPtree, myHeaderTab, minSup, set([]), myFreqList)
    return myFreqList

if __name__ == "__main__":
    rootNode = treeNode('pyramid', 9, None)
    rootNode.children['eye'] = treeNode('eye', 13, None)
    rootNode.children['phoenix'] = treeNode('phoenix', 3, None)
    # 将树以文本形式显示
    # print rootNode.disp()
    # load样本数据
    simpDat = loadSimpDat()
    # print simpDat, '\n'
    # frozen set 格式化 并 重新装载 样本数据，对所有的行进行统计求和，格式: {行：出现次数}
    initSet = createInitSet(simpDat)
    print(initSet)
    # 创建FP树
    # 输入：dist{行：出现次数}的样本数据  和  最小的支持度
    # 输出：最终的PF-tree，通过循环获取第一层的节点，然后每一层的节点进行递归的获取每一行的字节点，也就是分支。然后所谓的指针，就是后来的指向已存在的
    myFPtree, myHeaderTab = createTree(initSet, 3)
    myFPtree.disp()
    # 抽取条件模式基
    # 查询树节点的，频繁子项
    print('x --->', findPrefixPath('x', myHeaderTab['x'][1]))
    print('z --->', findPrefixPath('z', myHeaderTab['z'][1]))
    print('r --->', findPrefixPath('r', myHeaderTab['r'][1]))
    # 创建条件模式基
    freqItemList = []
    mineTree(myFPtree, myHeaderTab, 3, set([]), freqItemList)
    print(freqItemList)
    # 项目实战
    # 1.twitter项目案例
    # 无法运行，因为没发链接twitter
    lotsOtweets = getLotsOfTweets('RIMM')
    listOfTerms = mineTweets(lotsOtweets, 20)
    print(len(listOfTerms))
    for t in listOfTerms:
        print(t)

    # 2.新闻网站点击流中挖掘，例如：文章1阅读过的人，还阅读过什么？
    parsedDat = [line.split() for line in open('data/12.FPGrowth/kosarak.dat').readlines()]
    initSet = createInitSet(parsedDat)
    myFPtree, myHeaderTab = createTree(initSet, 100000)
    myFreList = []
    mineTree(myFPtree, myHeaderTab, 100000, set([]), myFreList)
    print(myFreList)

'''

'''
if frozenset(trans) not in retDict:   
            retDict[frozenset(trans)] = 1
        else:
            retDict[frozenset(trans)] += 1

'''
'''
问题解决：
1.for k in headerTable.keys():
RuntimeError: dictionary changed size during iteration
改为：
for k in list(headerTable.keys())   #将迭代对象转换成列表的形式

2.if not retDict.key(frozenset(trans)):
AttributeError: 'dict' object has no attribute 'key'
解决：
if frozenset(trans) not in retDict
'''



































