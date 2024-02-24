import pandas as pd
from pandas import set_option
from pandas.plotting import scatter_matrix
# Matplotlib is a comprehensive library for creating static, animated, and␣

import matplotlib.pyplot as plt
#sklearn Built on NumPy, SciPy, and matplotlib also used for data analysis
from sklearn.model_selection import train_test_split, KFold, cross_val_score

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

#Seaborn is a Python data visualization library based on matplotlib. It␣

import seaborn as sns
#numpy used to manipulate numerical data in python
import numpy as np
dataset = pd.read_csv('E:\BankNote_Authentication.csv')
print(dataset)
print('**********')
print(dataset.shape)
print('**********')
print(dataset.head(4))
print('**********')
print(dataset.describe())
print('**********')
print(dataset.groupby('entropy').size())
print('**********')
names = ['variance','skewness','curtosis','entropy','class']
# Pairwise pearson correlation
set_option('display.width', 100)
# 这个版本必须要display.precision才可以
set_option('display.precision', 3)  # 修改此处的 'precision' 为 'display.precision'
correlations = dataset.corr(method='pearson')
print(correlations)
print('')
skew = dataset.skew()
print(skew)

print()

# 获取数据集的列数
num_columns = len(dataset.columns)

# 计算适合的布局
num_rows = (num_columns + 1) // 2  # 加1是为了处理列数为奇数的情况
layout = (num_rows, 2)

# 绘制箱线图
dataset.plot(kind='box', subplots=True, layout=layout, sharex=False, sharey=False)
plt.show()

# 绘制直方图
dataset.hist(figsize=(10, 10))
plt.show()

scatter_matrix(dataset, figsize=(10,10))
plt.show()


