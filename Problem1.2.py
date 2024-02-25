import seaborn as sns
import pandas as pd
from matplotlib import pyplot as plt
import pandas as pd
from pandas import read_csv
from pandas import set_option
import matplotlib.pyplot as plt
import numpy as np
from numpy import set_printoptions
from sklearn.preprocessing import MinMaxScaler, Binarizer

# Load Basic Libraries
#Loading data and getting some info about data such as data type and dimensions
filename = 'BankNote_Authentication.csv'
dataset = read_csv(filename,  delimiter=',')
# Pairwise pearson correlation
set_option('display.width',100)
set_option('display.precision',3)
correlations=dataset.corr(method='pearson')
print(correlations)

 # Univariate Density Plot
dataset = read_csv(filename, delimiter=',')
dataset=dataset.drop(labels='skewness',axis=1)
dataset.plot(kind='density', subplots=True, layout=(5,7), sharex=False, figsize=(10,10))
plt.show()

#Correlations matrix plot
correlations=dataset.corr()
fig=plt.figure()
ax=fig.add_subplot(111)
cax=ax.matshow(correlations,vmin=-1,vmax=1)
fig.colorbar(cax)
plt.show()

print()
# Rescale data
array=dataset.values
X=array[:,0:2]
Y=array[:,2]
scaler=MinMaxScaler(feature_range=(0,1))
rescaledX=scaler.fit_transform(X)
set_printoptions(precision=3)
print(rescaledX[0:1,:])

print()
#Standardize data
from sklearn.preprocessing import StandardScaler
array=dataset.values
X=array[:,0:2]
Y=array[:,2]
scaler=StandardScaler().fit(X)
rescaledX=scaler.transform(X)
set_printoptions(precision=3)
print(rescaledX[0:1,:])

print()
#Normalize data
from sklearn.preprocessing import Normalizer
array=dataset.values
X=array[:,0:2]
Y=array[:,2]
scaler=Normalizer().fit(X)
normalizedX=scaler.transform(X)
set_printoptions(precision=3)
print(normalizedX[0:1,:])

print()
# Binarize data
array=dataset.values
X=array[:,0:2]
Y=array[:,2]
binarizer=Binarizer(threshold=0.0).fit(X)
binaryX=binarizer.transform(X)
set_printoptions(precision=3)
print(binaryX[0:1,:])