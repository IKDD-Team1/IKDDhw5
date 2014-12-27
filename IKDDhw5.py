# Date: 2014-12-21
# Author: brandboat(Kuan-Po Tseng)
# Description: for Kaggle Titanic
# References:
# (1) https://www.kaggle.com/c/titanic-gettingStarted/details/getting-started-with-python
# (2) https://www.kaggle.com/c/titanic-gettingStarted/details/getting-started-with-python-ii
# (3) http://corpocrat.com/2014/08/29/tutorial-titanic-dataset-machine-learning-for-kaggle/

import numpy as np
import pandas as pd
from sklearn import tree
from sklearn import cross_validation
from sklearn import ensemble

# read train.csv
df = pd.read_csv('./train.csv',header=0)

# cleanning data
# drop useless columns (Name, Ticket, Cabin)
cols = ['Name', 'Ticket', 'Cabin']
df = df.drop(cols,axis=1)

dummies = []
cols = ['Pclass','Sex','Embarked']
for col in cols:
    dummies.append(pd.get_dummies(df[col]))

titanic_dummies = pd.concat(dummies, axis=1)

df = pd.concat((df,titanic_dummies),axis=1)
df = df.drop(['Pclass','Sex','Embarked'],axis=1)
df['Age'] = df['Age'].interpolate()

x = df.values
y = df['Survived'].values
x = np.delete(x,1,axis=1)

# read train.csv
df = pd.read_csv('./test.csv',header=0)

# cleanning data
# drop useless columns (Name, Ticket, Cabin)
cols = ['Name', 'Ticket', 'Cabin']
df = df.drop(cols,axis=1)

dummies = []
cols = ['Pclass','Sex','Embarked']
for col in cols:
    dummies.append(pd.get_dummies(df[col]))

titanic_dummies = pd.concat(dummies, axis=1)

df = pd.concat((df,titanic_dummies),axis=1)
df = df.drop(['Pclass','Sex','Embarked'],axis=1)
df['Age'] = df['Age'].interpolate()
df['Fare'] = df['Fare'].interpolate()

x_results = df.values

clf = ensemble.GradientBoostingClassifier(n_estimators=50)
clf.fit(x,y)
y_results = clf.predict(x_results)
output = np.column_stack((x_results[:,0],y_results))
df_results = pd.DataFrame(output.astype('int'),columns=['PassengerID','Survived'])
df_results.to_csv('titanic_results.csv',index=False)
