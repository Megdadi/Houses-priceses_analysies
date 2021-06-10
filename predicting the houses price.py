#import libraries

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns # making statistical graphics in Python
import numpy as np
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

###############################bring in the six packs###################
df_train = pd.read_csv('D:/Houses-Regression/train.csv')
df_train.head()#Browse the first five lines
#check the columns
df_train.columns
#################################33
df_train['SalePrice']#price column display
df_train['SalePrice'].describe()#get the important details
##################################################
#histogram the price
sns.distplot(df_train['SalePrice'])
#######################################
#skewness and kurtosis
print("Skewness: %f" % df_train['SalePrice'].skew())# الانحراف
print("Kurtosis: %f" % df_train['SalePrice'].kurt())# فيشر
################ Relationship with numerical variables##############
#scatter plot grlivarea/saleprice
var = 'GrLivArea'# متغير المساحة
data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
# concat : وضع العمودين مع بعض
print(data[:5])
data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000))
##  It seems that 'SalePrice' and 'GrLivArea'
## have a really strong linear relationship
## And what about 'TotalBsmtSF'?
#scatter plot totalbsmtsf/saleprice
var = 'TotalBsmtSF'
data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000))
## this seems like a much strong linear (exponential )
## Moreover, it's clear that sometimes 'TotalBsmtSF' closes
## and gives zero improvement to 'SalePrice'.
####################Relationship with categorical features###################
#box plot overallqual/saleprice
var = 'OverallQual'
data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
f, ax = plt.subplots(figsize=(8, 6))
fig = sns.boxplot(x=var, y="SalePrice", data=data)
fig.axis(ymin=0, ymax=800000)
#########
#year built
var = 'YearBuilt'
data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
f, ax = plt.subplots(figsize=(16, 8))
fig = sns.boxplot(x=var, y="SalePrice", data=data)
fig.axis(ymin=0, ymax=800000);
plt.xticks(rotation=90)# rotate vertical the text axis 
## We note that is no strong tendency relation, but I think
## that the new staff has a priority to spend more money than old ancient.

################ Correlation matrix (heatmap style ####################
# correlation matrix between each two variables to take quick
# overview of its relationships
# the heatmap can not get the categorical features
corrmat = df_train.corr()
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat, vmax=.8, square=True)
# the last column (SalePrice) is the most important

##################################
# Now, we want to focus on the strong Correlation
# we will remove the variables (black color) that don't effect on SalePrice
#saleprice correlation matrix
k = 11 #number of variables for heatmap
cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index# pick the best powerfull Correlation(more 51%)
cm = np.corrcoef(df_train[cols].values.T)
sns.set(font_scale=1.25)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
plt.show()
###########scatterplot#############
sns.set()
cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']
sns.pairplot(df_train[cols], size = 2.5)
plt.show()

############# Missing data #####################
# total = df_train.isnull()
# print(total)
# print (len(total))
total = df_train.isnull().sum().sort_values(ascending=False)
percent = (df_train.isnull().sum()/df_train.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data.head(20)
## but, how to handle the missing data?
# we will remove the features that have total(null nalue) > 7

df_train = df_train.drop((missing_data[missing_data['Total'] > 82]).index,1)
df_train.isnull().sum().max()
#################################################33

