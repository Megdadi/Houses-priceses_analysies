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

missing_values= df_train.isnull().sum().sort_values(ascending=False)
percent=((missing_values/df_train.isnull().count()) *100).sort_values(ascending=False)
missingdata_percent=pd.concat([missing_values, percent], axis=1, keys=['Total', 'Percent'])

feature_has_bigget_null= missingdata_percent[missingdata_percent['Total'] > 82]
train_after_delet_nulls=df_train.drop(feature_has_bigget_null.index,1)
print(train_after_delet_nulls.shape) # (1460, 75)

train= train_after_delet_nulls
train.describe()
train['SalePrice'].describe()

################### Anomaly Detection (Outliers) #################
############# ex: GrLivArea & SalePrice

plt.figure(figsize=(9,6))
var = 'GrLivArea'# متغير المساحة
data = pd.concat([train['SalePrice'], train[var]], axis=1)
sns.scatterplot(x='SalePrice',y='GrLivArea',data=train)
# We note that they are two outlier points
# delete two points
a=train[train['GrLivArea'] > 4000].index
b=train[train['SalePrice'] > 700000].index
points= a&b
train = train.drop(points)

#Check the graphic again
plt.figure(figsize=(9,6))
var = 'GrLivArea'# متغير المساحة
data = pd.concat([train['SalePrice'], train[var]], axis=1)
sns.scatterplot(x='SalePrice',y='GrLivArea',data=train)

############# OverallQual & SalePrice

plt.figure(figsize=(9,6))
var = 'OverallQual'# Quality
data = pd.concat([train['SalePrice'], train[var]], axis=1)
sns.scatterplot(x='SalePrice',y='OverallQual',data=train)
print(data[:5])
# We note that they are not outlier points 

############# YearBuilt & SalePrice

plt.figure(figsize=(9,6))
var = 'YearBuilt'# تاريخ البناء
data = pd.concat([train['SalePrice'], train[var]], axis=1)
sns.scatterplot(x='SalePrice',y='YearBuilt',data=train)
# We note that is no strong tendency relation, but I think
#that the new staff has a priority to spend more money than old ancient

############# TotalBsmtSF & SalePrice

plt.figure(figsize=(9,6))
var = 'TotalBsmtSF'# تاريخ البناء
data = pd.concat([train['SalePrice'], train[var]], axis=1)
sns.scatterplot(x='SalePrice',y='TotalBsmtSF',data=train)


