import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

dataset = pd.read_csv('../../datasets/studentscores.csv')
X = dataset.iloc[:, : 1].values
Y = dataset.iloc[:, 1].values

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=0)

print 'X_train = %s' % X_train
print 'Y_train = %s' % Y_train
print 'X_test = %s' % X_test
print 'Y_test = %s' % Y_test

# https://www.jianshu.com/p/64b1404faaa4
regressor = LinearRegression()
regressor = regressor.fit(X_train, Y_train)

Y_pred = regressor.predict(X_test)
print 'Y_pred = %s' % Y_pred

plt.scatter(X_train, Y_train, color='red')
plt.plot(X_train, regressor.predict(X_train), color='blue')
plt.show()

plt.scatter(X_test, Y_test, color='red')
plt.plot(X_test, regressor.predict(X_test), color='blue')
plt.show()