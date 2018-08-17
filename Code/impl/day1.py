import warnings

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Imputer, LabelEncoder, OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore", message="numpy.dtype size changed")

dataset = pd.read_csv('../../datasets/Data.csv')

X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, 3].values

print 'row data'
print X
print Y

imputer = Imputer(missing_values="NaN", strategy="mean", axis=0)
imputer = imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])

print 'completed data'
print X

# enum -> label [male, female] => [0, 1]
labelencoder_X = LabelEncoder()
X[:, 0] = labelencoder_X.fit_transform(X[:, 0])

print 'label country'
print X

# https://blog.csdn.net/bitcarmanlee/article/details/51472816
onehotencoder = OneHotEncoder(categorical_features=[0])
onehotencoder.fit(X)
print 'n_values is: ', onehotencoder.n_values_
print 'feature_indices is', onehotencoder.feature_indices_
X = onehotencoder.transform(X).toarray()

print 'one hot country'
print X

labelencoder_Y = LabelEncoder()
Y = labelencoder_Y.fit_transform(Y)

print 'label result'
print Y

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

print 'train data'
print X_train
print Y_train

print 'test data'
print X_test
print Y_test

# https://blog.csdn.net/u012609509/article/details/78554709
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.fit_transform(X_test)


print 'standardized train data'
print 'train data'
print X_train
print 'test data'
print X_test