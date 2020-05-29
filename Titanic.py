import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

titanic = pd.read_csv('/Users/a./Downloads/titanic.csv')
titanic['Sex'].replace('female', '0',inplace=True)
titanic['Sex'].replace('male', '1',inplace=True)
titanic.Age.fillna(value=titanic.Age.mean(), inplace=True)
titanic.Fare.fillna(value=titanic.Fare.mean(), inplace=True)
titanic.Embarked.fillna(value=(titanic.Embarked.value_counts().idxmax()), inplace=True)
titanic.Survived.fillna(value=-1, inplace=True)
titanic.Sex.fillna(value=0, inplace=True)
titanic.Pclass.fillna(value=titanic.Pclass.mean(), inplace=True)

stats=[]
print("Hey There? Do you think You'd Survive on the Titanic ?? ")
x = input("Whats ur gender: 1/M,0/F ")
y = input("What class r u travelling? 1/2/3 ")
z = input("Whats ur AGE?? ")

stats.append(y)
stats.append(x)
stats.append(z)
numpy = []
numpy.append(stats)
numpy = np.array(numpy)

X= np.array(titanic[['Pclass','Sex','Age']])
Y = np.array(titanic[['Survived']])

X_train,X_test,y_train,y_test = train_test_split(X,Y,test_size = 0.2)
knn = KNeighborsRegressor(1).fit(X_train,y_train)
linreg = LinearRegression().fit(X_train,y_train)
kcc = KNeighborsClassifier(1).fit(X_train,y_train)
acc = (kcc.score(X_test,y_test))
Jack = kcc.predict(numpy)
print("1 Means Yes, 0 Means NO ! ",Jack)
print("The result is being reported with ", acc ," Accuracy")
