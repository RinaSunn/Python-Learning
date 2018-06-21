#! /usr/bin/env python3.6

import time
import numpy as np
import pandas as pd
import xlrd

### define function "localMaxima":

vlist = [1,0,1,1,0,1]
vlist[vlist==1]


################ 0. 函数的声明       ################
################ defining funciton  ################


#继续上周的小练习，我们写一个斐波那契 函数
#函数功能是take 一个数值型input "n"，然后返回斐波那契数列中第 "n"位的元素。

def fibonacci(n):
    lfibo = []
    for i in range(n):
        if len(lfibo) >= 2:
            lfibo.append(sum([lfibo[i - 2], lfibo[i - 1]]))
        else:
            lfibo.append(1)
            ### [list].append() 列表自带方法 append(), 在列表后加入一个新的元素。
            # 在这里我们可以看到，这里并没有对lfibo 这个变量名进行任何重新赋值。
            # 因为lfibo.append()是直接针对于列表对象进行的操作。
    nth = lfibo.pop()
    ### [list].pop() 列表类自带方法 pop(), 将数列最后一个进入的数值从列表中剔除，并且返回该值。
    return nth
    ### 这个return 在这里表示fibonacci()函数会在最终返回nth这个变量的值。我们也可以直接 return lfibo.pop()



### 写一个函数，通过一个给定的含有数值型元素的[list]，返回其中最大的值/值的index
def maxima(lnum):
    maxima = lnum[0]
    for i in range(len(lnum)):
        if lnum[i]>maxima:
            maxima = lnum[i]
    return maxima

def maxima_pos(lnum):
    maxima_pos = 0
    for i in range(len(lnum)):
        if lnum[i]>lnum[maxima_pos]:
            maxima_pos = i
    return maxima_pos



################ 1. numpy.array & pandas.series ################

colname = []
strength = pd.Series([95,65,25,45])

employee = {"fam_name":["Sun","Wang","Wang","Yu"],
            "given_name": ["Ruixiaotong","Linna","Cheng","Weijian"],
            "age":[18,18,27,26],
            "gender":["female",'female','male','male'],
            "height":[168,168,191,183]}

empdf = pd.DataFrame(data= employee)
#### 如何filter,比方说我们想要找到 age <=18 的员工
empU18 = empdf[empdf.age <= 18]
###如何理解以上语句：
###看一下每一个环节都做了什么/ 返回了什么值
print(empdf.age)
### empdf.age 返回了variable_name = 'age'的这一列数据，也就是pandas.Series 一维数组。
# 就像我们讲的，pandas.series包含两个信息，一列是数组的数值，一列是数组的对应位置 index.

print(empdf.age <= 18)
### numpy.array还有 pandas.Series 与 python自带的list最大的区别，就是前者可以很方便的使用 布尔型索引。
### empdf[[True,True,False,True]],其中的boolean list的长度要和 Dataframe的row数量一致


### 如果我们想得到18或以下年龄的员工姓名组成的列表 我们应该怎么做
### 以下是利用pandas Dataframe Object和 Series Object的类方法
nameU18 = []
for i in empdf.iterrows():
    if i[1].age <=18:
        nameU18.append(i[1].fam_name + " " + i[1].given_name)


### 以下是 花式操作 # List Comprehension
nameU18 = [i[1].fam_name + " " + i[1].given_name for i in empdf.iterrows() if i[1].age <=18]


#对dataframe进行基本修改 （查删改）
employee = pd.DataFrame([["ID","Age","Salary","Height","Gender"],[1,27,14,178,"F"],
                         [2,28,np.nan,169,"F"],[3,27,9,191,"M"],[4,26,9,183,"M"],[5,24,11,170,"F"]])
#去掉第一行，并将这一行定义为df的column name
colname = employee.ix[0]
employee = employee.ix[1:]
employee.columns = colname

#转置矩阵
employeeTrans = employee.T



############ 从这里往下我们用从网上找来的数据，尝试做一个niche analysis ############
#### 背景： 从网上得到了一个加州犯罪统计，每种刑法定义罪名都有统计数据。
#### 补充数据   1.wikipedia中搜索得到的加州最近一个财政年的：按地区分类的 收入统计/排名
####           2. wikipedia中搜索得到的加州最近一个财政年统计的：社区组成，按race分区
#### 时效性：由于是mock practice， 数据取得自不同时间段，没有时效性，纯粹for the sake of practice
#### population 以 crime analysis.xlsx中记录为主。 不考虑另外两份补充数据中 人口数据。只考虑占比数据。

# 导入模块
import pandas as pd
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import seaborn
from sklearn import linear_model
import statsmodels.api as sm
from sklearn.linear_model import Lasso

# 声明数据路径
# 这些源文件我有简单进行清理，但是清理任务没有完全完成
path = "C:\\Users\\c.wang1\\Desktop\\Practices\\"
crime_path = "crime analysis.xlsx"
income_path = "city income.xlsx"
race_path = "Racial Composition.xlsx"


# 读入文件
crime = pd.read_excel(path+crime_path)
income = pd.read_excel(path+income_path)
race = pd.read_excel(path+race_path)

# 整理文件列名
#1.
tmpname = {'City name':'cityName','Population':'population','Crime index (All crimes except arson)':'crimeIndex',
           'Total crime index (All crimes including arson)':'totalCrimeIndex',
           'Murder and nonnegligent manslaughter':'murder','Forcible rape':'rape','Robbery':'robbery',
           'Aggravated assault':'assault','Burglary':'burglary','Larceny-theft':'larcenyTheft',
           'Motor vehicle theft':'GTA','Arson':'arson'}
crime = crime.rename(columns = tmpname)
#2. 为了阅读方便，以及传阅代码时格式不会乱行，请将过长代码进行换行
tmpname = {'Place':'cityName', 'County':'county', 'Population':'population','Population Density':'populationDensity','Per capita income':'perCapitaIncome','Median household income':'medianHouseholdIncome', 'Median family income':'medianFamilyIncome'}
income = income.rename(columns = tmpname)
#3.
tmpname = {'Place':'cityName','County':'county', 'Population':'population','White':'white','Other':'other',
           'Asian':'asian','Black or African':'black','Native American':'nativeAmerican',
           'Hispanic or Latino':'latino'}
race = race.rename(columns = tmpname)
# 看某些stackoverflow,里面推荐答案，重命名colname使用语句: df.columns = [list of new name]，
# 实际这个方法是错误的，这只是建立了一个链接，
# 而不是直接改变了Dataframe的object属性。如果要改变columns属性，
# 则需要用上述方法 df=  df.rename(columns = {one-one dict of new name})



# data cleaning
#1.对于 income来说，去掉空白行（由于excel merge造成的）
#利用布尔检索，去掉NA值
income = income[-income.cityName.isna()]
race = race[- race.cityName.isna()]

#2.对三个表格进行sort，均按cityName sort，为merge做准备

crime = crime.sort_values("cityName")
income = income.sort_values("cityName")
race = race.sort_values("cityName")

#3. 对于在三个表格中都出现的城市记录进行merge
#   首先要将主键改为全大写，以免出现由于大小写导致的无法对齐

crime.cityName = crime.cityName.str.upper()
income.cityName = income.cityName.str.upper()
race.cityName = race.cityName.str.upper()



### 完成这一步之后发现矩阵的大小变化为
### crime: [335:12], ri: [332:18], cri[342:26]
### 这说明主键出现过多次。 我们需要按主键除重
ri = race.merge(income,left_on = 'cityName',right_on = 'cityName')
cri = crime.merge(ri,left_on = 'cityName',right_on = 'cityName')


# 除重, principle ：remove all duplicate： 重复出现的主键全部删除，我们只观察出现过一次的city。
# 最终得到 df: [322:26] 的矩阵
df = cri.drop_duplicates(subset= 'cityName', keep = False)
# 返回一个removed城市列表留档供参考。
cityRemoved =cri[cri.duplicated(subset = 'cityName')].cityName.drop_duplicates()

###我们还发现由于wikipedia加的reference有个 '[7]'的角标，而在income 表格中，有一些元素值由于是unavaliable被加了角标，
### 而导致最终这一列数据被定义为 object 型，而不是numeric型。为了往后的计算方便，我们需要对这些元素值进行更改。
#具体操作：
# 我们只需要将这一行选定，然后选定这两列，进行更改即可。
df.loc[df.medianFamilyIncome =='[7]',['medianFamilyIncome','medianHouseholdIncome']] = [np.nan,np.nan]
df.loc[:,'medianFamilyIncome'] = pd.to_numeric(df.medianFamilyIncome)
df.loc[:,'medianHouseholdIncome'] = pd.to_numeric(df.medianHouseholdIncome)
df.loc[:,'perCapitaIncome'] = pd.to_numeric(df.perCapitaIncome)


#### 数据清理/预处理结束 ####
### 收入与种族占比模型
### 首先去冗余， 检查收入indicative变量corr
df=df.drop(['county_x','county_y','population_x','population_y'],axis = 1)

### 生成一个heatmap检查各项numeric 数据相关性
### 调用seaborn.heatmap功能，对df中 dtype为numeric的列数据的关联矩阵 进行heatmap生成。
is_numeric = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
seaborn.heatmap(df.select_dtypes(include= is_numeric).corr(),cmap = 'PiYG',center=0)


### 将包含racial composition的信息的列，以及 包含犯罪程度的列，分别列出来
raceCol = ['white','asian','latino','black','other','nativeAmerican']
incomeCol = ['perCapitaIncome','medianHouseholdIncome','medianFamilyIncome','populationDensity']
crimeCol = ['cityName', 'population', 'crimeIndex', 'totalCrimeIndex',
       'murder', 'rape', 'robbery', 'assault', 'burglary', 'larcenyTheft',
       'GTA', 'arson']


### 查看有色人种和 犯罪数量占人口比例 间关系
check = ['crimeIndex','black','white','latino','other','nativeAmerican','asian']
dfrace = df[check]
seaborn.heatmap(dfrace.select_dtypes(include= is_numeric).corr(),cmap = 'coolwarm',center=0)

### 计算/插入 按人种划分的人口数量
df['black_population']= np.floor(df.population * df.black)
for race in raceCol:
    df["{0}_population".format(race)] = np.floor(df.population * df[race])



#from sklearn.linear_model import Lasso
predictors = []
predictors.extend(raceCol)
predictors.extend(incomeCol)
df1 = df.dropna()
lasso.fit(df1[predictors],df1.crimeIndex)
lasso_coef = lasso.coef_

plt.clf()
plt.cla()
plt.close()
plt.plot(range(len(predictors)),lasso_coef)
plt.xticks(range(len(predictors)),df1[predictors].values,rotation = 60)
plt.margins(0.02)






#def lasso_regression(data,x,y,alpha,models_to_plot = {}):
#    #fit model
#    lassoreg= Lasso(alpha = alpha, normalize = True, max_iter= 1e3)
#    lassoreg.fit(data[x],data[y])
#    y_pred = lassoreg.predict(data[x])
#
#    if alpha in models_to_plot:
#        plt.subplot(models_to_plot[alpha])
#        plt.tight_layout()
#        plt.plot(data['population'],y_pred)
#        plot.plot(data['population'],data[y],'.')
#
#    rss = sum((y_pred-data['y'])**2)
#    ret = [rss]
#    ret.extend([lassoreg.intercept_])
#    ret.extend(lassoreg.coef_)
#    return ret


#### 加权范围
#alpha_lasso = [1e-15,1e-10,1e-8,1e-5,1e-4,1e-3,1e-2,1,5,10]

#col = ['rss','intercept'] + ['coef_%i' for i in predictors]
