# Multiple Linear Regression after Backward Elimination

import numpy as np 
import pandas as pd
# Importing the dataset
df=pd.read_csv('C:\\Users\\שני שלום\\Desktop\\סמסטר ב\\Machine Learning\\עבודה\\autos.csv', na_values=('?'))

# Drop rows that don't have a price:
df = df.dropna(subset=['price'])

# Delete the 'engine-location' column:
df = df.drop(columns=['engine-location'])

# Delete the 'fuel-system' column:
df = df.drop(columns=['fuel-system'])

# Replace the numeric names in the categorical column with its digits:
df['num-of-doors'] = df['num-of-doors'].replace(('two','four'),(2,4))
df['cylinders'] = df['cylinders'].replace(('two','three','four','five','six','eight','twelve'),(2,3,4,5,6,8,12))

X = df.iloc[:,:-1].values
Y = df.iloc[:,-1].values

from sklearn.impute import SimpleImputer
# Handling missing numeric data in several columns:
imp_median = SimpleImputer(missing_values=np.nan, strategy='median')
X[:,[1,15,16,18,19]]=imp_median.fit_transform(X[:,[1,15,16,18,19]])
# Handling missing data in column 'num-of-doors':
imp_most_frequent = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
X[:,[4]]=imp_most_frequent.fit_transform(X[:,[4]])
#Y = Y.reshape(201, 1)

# Encoding categorical columns:
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:,2] = labelencoder_X.fit_transform(X[:,2])
onehotencoder = OneHotEncoder(sparse=False)
A= onehotencoder.fit_transform(X[:, [2]])
X = np.hstack(( A, X[:,:2] , X[:,3:]))
X = X[:, 1:]
B= onehotencoder.fit_transform(X[:, [3]])
X = np.hstack(( B, X[:,:3] , X[:,4:]))
X = X[:, 1:]
C= onehotencoder.fit_transform(X[:, [5]])
X = np.hstack(( C, X[:,:5] , X[:,6:]))
X = X[:, 1:]
D= onehotencoder.fit_transform(X[:, [9]])
X = np.hstack(( D, X[:,:9] , X[:,10:]))
X = X[:, 1:]
E= onehotencoder.fit_transform(X[:, [16]])
X = np.hstack(( E, X[:,:16] , X[:,17:])).astype('float')
X = X[:, 1:]

# Backwards elimination
import pandas.util.testing as tm
import statsmodels.tools.tools as tl
X = tl.add_constant(X)
import statsmodels.api as sm
X=X[ :, [0,2,3,4,5,7,8,9,10,11,13,19,20,23,24,25,26,27,28,30]]
regressor_OLS=sm.OLS(endog=Y, exog=X).fit()
regressor_OLS.summary()

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)

# Fitting Multiple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, Y_train)

# Predicting the Test set results
Y_pred = regressor.predict(X_test) #Compare to y_test

# Residuals calculation
residuals=np.average(np.abs(Y_pred-Y_test))
print(residuals)
















