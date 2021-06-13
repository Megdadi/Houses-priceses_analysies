
import pandas as pd ## data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # Matlab-style plotting
import seaborn as sns # making statistical graphics in Python
import numpy as np # linear algebra
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

df_train = pd.read_csv('D:/Houses-Regression/train.csv')



############# Handling missing data #####################
print(df_train.shape) # (1460, 81)

missing_values= df_train.isnull().sum().sort_values(ascending=False)
percent=((missing_values/df_train.isnull().count()) *100).sort_values(ascending=False)
missingdata_percent=pd.concat([missing_values, percent], axis=1, keys=['Total', 'Percent'])

feature_has_bigget_null= missingdata_percent[missingdata_percent['Total'] > 82]
train_after_delet_nulls=df_train.drop(feature_has_bigget_null.index,1)
print(train_after_delet_nulls.shape) # (1460, 75)