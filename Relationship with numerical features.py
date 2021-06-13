
import pandas as pd ## data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # Matlab-style plotting
import seaborn as sns # making statistical graphics in Python
import numpy as np # linear algebra
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


############# Handling missing data #####################

df_train = pd.read_csv('D:/Houses-Regression/train.csv')

print(df_train.shape) # (1460, 81)

# missing_values= df_train.isnull().sum().sort_values(ascending=False)
# percent=((missing_values/df_train.isnull().count()) *100).sort_values(ascending=False)
# missingdata_percent=pd.concat([missing_values, percent], axis=1, keys=['Total', 'Percent'])

# feature_has_bigget_null= missingdata_percent[missingdata_percent['Total'] > 82]
# train_after_delet_nulls=df_train.drop(feature_has_bigget_null.index,1)
# print(train_after_delet_nulls.shape) # (1460, 75)

# train= train_after_delet_nulls
# train.describe()
# train['SalePrice'].describe()


################ Relationship with numerical variables##############
#scatter plot grlivarea/saleprice
var = 'GrLivArea'# متغير المساحة
data = pd.concat([train['SalePrice'], train[var]], axis=1)
# concat : وضع العمودين مع بعض
print(data[:5])
data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000))
##  It seems that 'SalePrice' and 'GrLivArea'
## have a really strong linear relationship
## And what about 'TotalBsmtSF'?
#scatter plot totalbsmtsf/saleprice
var = 'TotalBsmtSF'
data = pd.concat([train['SalePrice'], train[var]], axis=1)
data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000))
## this seems like a much strong linear (exponential )
## Moreover, it's clear that sometimes 'TotalBsmtSF' closes
## and gives zero improvement to 'SalePrice'.
####################Relationship with categorical features###################
#box plot overallqual/saleprice
var = 'OverallQual'
data = pd.concat([train['SalePrice'], train[var]], axis=1)
f, ax = plt.subplots(figsize=(8, 6))
fig = sns.boxplot(x=var, y="SalePrice", data=data)
fig.axis(ymin=0, ymax=800000)
#########
#year built
var = 'YearBuilt'
data = pd.concat([train['SalePrice'], train[var]], axis=1)
f, ax = plt.subplots(figsize=(16, 8))
fig = sns.boxplot(x=var, y="SalePrice", data=data)
fig.axis(ymin=0, ymax=800000);
plt.xticks(rotation=90)# rotate vertical the text axis 
## We note that is no strong tendency relation, but I think
## that the new staff has a priority to spend more money than old ancient.