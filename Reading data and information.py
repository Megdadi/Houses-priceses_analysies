
import pandas as pd ## data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # Matlab-style plotting
import seaborn as sns # making statistical graphics in Python
import numpy as np # linear algebra
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from scipy import stats
import warnings
warnings.filterwarnings('ignore')
################################
df_train = pd.read_csv('D:/Houses-Regression/train.csv')
types=df_train.dtypes
firsr_five_rows=df_train.head()
last_five_rows=df_train.tail()
statistical_details= df_train.describe()#  basic statistical details like percentile, mean
statistical_details_SalePrice = df_train['SalePrice'].describe()
df_train.columns #check the columns (features)
df_train['SalePrice']#price column display
df_train.columns

#skewness and kurtosis
print("Skewness: %f" % df_train['SalePrice'].skew())# الانحراف
print("Kurtosis: %f" % df_train['SalePrice'].kurt())# فيشر

plt.figure(figsize=(8,5))
sns.distplot(df_train['SalePrice']) #histogram the price
sns.countplot(df_train['SalePrice'])
mean = df_train['SalePrice'].mean(axis=0)
std = df_train['SalePrice'].std(axis=0)
print(df_train['SalePrice'].mean(axis=0))
print(df_train['SalePrice'].std(axis=0))

concise_summary= df_train.info() # concise summary for dtype and column dtypes, non-null values and memory usage

#check the numbers of samples and features
print("The train data size before dropping Id feature is : {} ".format(df_train.shape))# (1460, 81)
