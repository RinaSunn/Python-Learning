#打印100以内的斐波那契数列
#斐波那契数列可以以递归的方式定义：F0=0， F1=1， Fn=Fn-1+Fn-2
#用文字来说就是：斐波那契数列由0和1开始，之后的斐波那契数列系数就是由之前的两数相加。

>>> fibo=list(range(2))
>>> for i in range(2,101):
    fibo.append(sum([fibo[i-1],fibo[i-2]]))
print(fibo)


### Exercise:
### Answer Q1: fibonacci

fibonacci = []
length = 100
for i in range(length):
    if len(fibonacci)>=2:
        fibonacci.append(sum([fibonacci[i-2],fibonacci[i-1]]))
    else:
        fibonacci.append(1)


###这个自己解答不会
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

###这个自己解答不会
#Answer Q3: play with list index and string.format()
# "string".format() 字符串[类方法] 的运用。 python字符串自带class method:  .format().
# "My name is {0}, I'm {1} years old, I was {2} 5 years ago.".format("Cheng",27,27)
# 这其中 {数字} 会被 format()中对应位置的值替代。 format中的值可以为任意型，也可以为函数/语句。
vlist = [1,2,3,4,5,6]
for n in range(len(vlist)):
    print("at pos: {0},\nvlist element is {1},\nthe cumulative sum is {2}".format(n,vlist[n],sum(vlist[:n+1])))