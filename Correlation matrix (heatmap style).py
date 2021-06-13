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

################ Correlation matrix (heatmap style) ####################
# correlation matrix between each two variables to take quick
# overview of its relationships
# the heatmap can not get the categorical features
corrmat = train.corr()
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat, vmax=.8, square=True)
# the last column (SalePrice) is the most important
################################################
# Now, we want to focus on the strong Correlation
# we will remove the features that don't effect on SalePrice
#saleprice correlation matrix
k = 11 #number of variables for heatmap
cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index# pick the best powerfull Correlation(more 51%)
cm = np.corrcoef(train[cols].values.T)
# 
sns.set(font_scale=1.25)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
plt.show()

###########scatterplot#############
sns.set()
aa=[cols]
power_feature = np.array(aa)# convert list to array
for x in power_feature:
    sns.pairplot(train[x], size = 2.5)
    plt.show()
    

#######################################################

power_featues_train= train[cols]




