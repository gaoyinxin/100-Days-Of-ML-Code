import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

dataset = pd.read_csv('../../datasets/50_Startups.csv')
X = dataset.iloc[:, : -1].values
Y = dataset.iloc[:, 4].values

labelencoder = LabelEncoder()
X[:, 3] = labelencoder.fit_transform(X[:, 3])
print 'X = %s' % X

onehotencoder = OneHotEncoder(categorical_features=[3])
X = onehotencoder.fit_transform(X).toarray()
print 'X = %s' % X

# Dummy variable trap
X = X[:, 1:]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20, random_state=0)

print 'X_train = %s  (%d)' % (X_train, len(X_train))
print 'Y_train = %s  (%d)' % (Y_train, len(Y_train))
print 'X_test = %s  (%d)' % (X_test, len(X_test))
print 'Y_test = %s  (%d)' % (Y_test, len(Y_test))

# https://www.jianshu.com/p/64b1404faaa4
regressor = LinearRegression()
regressor = regressor.fit(X_train, Y_train)

print 'intercept %d' % regressor.intercept_
print 'coef %s' % regressor.coef_

Y_pred = regressor.predict(X_test)
print 'Y_pred = %s' % Y_pred

# https://www.cnblogs.com/pinard/p/6016029.html
print "MSE:", metrics.mean_squared_error(Y_test, Y_pred)
print "RMSE:", np.sqrt(metrics.mean_squared_error(Y_test, Y_pred))
