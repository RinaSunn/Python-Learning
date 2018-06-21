w.import time
import numpy as np


############### 0. Variable Assignment ################

sentence = "Hello World"                    #string

number = 1                                  #numeric

vBoolean = True                             #boolean

vList = []                                  #object
vList = [1,2,3,4,5]
vTuple = ()
vTuple = (1,2,3,4,5)
vDict = {}
vDict ={"a":1,"c":3, "b": True,"d":vList}




############### 1. Function Execution ################
print("Hello World")

a = 3
b = 2
number = 2**3
yi = "1"
er = "2"
shuzi = "1+2"
sentence = "Hello World"
print(sentence)
print(number)
print(number)
print(shuzi)
print("hello world")
print(1)
print(yi+number)



############### 2. Random Indexing, Mutable & Immutable ###############
#### data type:
#initiate:
lfruit = ['apple','banana','pear','carrot']
tfruit = ('apple','banana','pear','carrot')

#Random Indexing
print(tfruit[0])
print(lfruit[2])
print(lfruit[-1])

#mutable
lfruit[3] = "cherry"

#immutable
tfruit[3] = "pinnapple"

#list.method
lfruit.pop()
lfruit.pop(0)
lfruit.remove("apple")
lfruit.append("watermelon")

###返回ppt.下面内容为逻辑控制
################ 3. Flow Control & Loop ################
# if...else
switch = "OFF"
if switch == "ON":
    print("The light is \"On\"")
elif switch =="OFF":
    print("The light is \"OFF\"")
print(switch=="ON")

# Loop:
# For..... (do) ....
# While... (do)....
"""
Java:
    for(int i=0; i<5; i++){
        System.out.println("i is:" + i);
    }
"""
letters = ('a','b','c','d')
letters_list = ['a','b','c','d']

for i in range(5):
    print(i)
for j in letters:
    print(j)
#用 for loop 打印 1-10的数：
for i in range(1,11):
    print(i)
#用 while loop 打印 1-10的数：
i= 1
while i <= 10:
    print(i)
    i= i+1  #i+=1

start = 5
#打印 start - 10 之间的数
for i in range(start,11):
    print(i)
while start <=10:
    print(start)
    start +=1


###\t indent实例:
#遍历 3x3 乘法表所有结果：
for i in range(1,4):
    for j in range(1,4):
        a= i*j
        print(a)

for i in range(1,4):
    for j in range(1,4):
        a= i*j
    print(a)
### 不同的缩进 代表了print()函数执行的时间点不同

# 小练习
# 寻找100以内完全平方数
# 100以内开方为整数的数：
for i in range(1,101):
    if (i**0.5)%1 ==0:
        print(i)

# 寻找789 - 889 以内的完全平方数
start = 789
end = 889
for i in range(start,end+1):
    if (i**0.5)%1 ==0:
        print(i)










### Exercise:
### Answer Q1: fibonacci

fibonacci = []
length = 100
for i in range(length):
    if len(fibonacci)>=2:
        fibonacci.append(sum([fibonacci[i-2],fibonacci[i-1]]))
    else:
        fibonacci.append(1)

### Answer Q2: finding prime number
a = 1
prime_list  = []
while a <= 100:
    is_prime = True
    for i in range(2,a//2):
        if a/i%1==0:
            is_prime = False
            break
    if is_prime:
        prime_list.append(a)
    a+=1

#Answer Q3: play with list index and string.format()
# "string".format() 字符串[类方法] 的运用。 python字符串自带class method:  .format().
# "My name is {0}, I'm {1} years old, I was {2} 5 years ago.".format("Cheng",27,27)
# 这其中 {数字} 会被 format()中对应位置的值替代。 format中的值可以为任意型，也可以为函数/语句。
vlist = [1,2,3,4,5,6]
for n in range(len(vlist)):
    print("at pos: {0},\nvlist element is {1},\nthe cumulative sum is {2}".format(n,vlist[n],sum(vlist[:n+1])))