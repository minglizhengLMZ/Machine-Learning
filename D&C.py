# -*- coding: utf-8 -*-
"""
Created on Sat Nov  2 10:34:01 2019

@author: Administrator
"""
'''-----使用循环求和------'''
def sum(arr):
    total = 0
    for i in arr:
        total +=i
    return total

sum([1,2,3,4])

'''-------使用递归函数求和------'''
#（1）确定基线条件  
#（2）确定递归条件：每次调用后都可以缩小问题的规模
def sum(list):
    if list ==[]:     #设定基线条件（思路，最简单的问题--此处为空列表）
        return 0 
    else:
        return list[0]+sum(list[1:])  #递归条件，迭代本事，问题不断简化
sum([1,2,3,4])        

'''----使用递归计算列表元素数--------'''
def  count(list):
    if list == []:
        return 0
    else:
        return 1+count(list[1:])
    
count([1,2,3,4,5])

#查找列表的最大数字
def maxnu(list):
    if len(list)==0:     #基线条件
        return None
    if len(list) ==1:
        return list[0]
    else:
        sub_maxnu=maxnu(list[1:])   #递归条件  （先逐步调用再倒着依次返回）
        return list[0] if list[0]> sub_maxnu  else sub_maxnu
    
maxnu([5,6,4,3,1,7,2]) 

'''--------------------------快速排序(其性能依赖于设定的基准值)-----------------'''   
def quicksort(array):
    if len(array)<2:   #基线条件，数组仅一个或哦个元素时无需排序
        return array
    else:                
#循环条件，设定基准值，将数组根据基准值划分为两部分，再依次对两部分进行快速排序
        pivot = array[0]    #
        less = [i for i in array[1:]if i<=pivot]
        greater = [i for i in array[1:] if i>pivot]
        return quicksort(less) + [pivot] + quicksort(greater)

print(quicksort([1,4,9,3,6,2,8]))

'''----------------大O表示法---------'''
def print_items(list):
    for item in list:
        print (item)
prin        
#每次打印前都设定一次休眠
from time import sleep
def print_items2(list):
    for item in list:
        sleep(1)
        print(item)

print_items([2,4,5,6,8,0])
print_items2([2,4,5,6,8,0])
    
    