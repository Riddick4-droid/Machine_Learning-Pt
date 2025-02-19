import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import style
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
from sklearn.model_selection import train_test_split
style.use('fivethirtyeight')


"""importing data into script"""
data = pd.read_excel('buys_computer.xlsx')
print(data.head(10))

"""data cleaning"""
"""This includes removing irrelevant columns"""
"""and null cells"""
print(data.info())

"""the data.info() command shows us the entire"""
"""properties of the dataset we have. It allows us to """
"""view the columns and tell which one have null values"""

"""label encoder will convert all columns that
need to be in digits or values for our work"""
le = LabelEncoder()

data['income'] = le.fit_transform(data['income'])##where 0 is high, 1 is low and medium is 2
data['student'] = le.fit_transform(data['student'])##where 0  for no and 1 for yes
data['credit_rating'] = le.fit_transform(data['credit_rating'])##where 1 is for fair and 0 is for excellent
data['buys_computer'] = le.fit_transform(data['buys_computer'])##where 0 is no and 1 is yes
data['age'] = le.fit_transform(data['age'])
print(data.head())

"""feature selection/engineering"""
y = data['buys_computer']
X = data.drop('buys_computer', axis=1)


"""train_test_split data"""
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state = 42)

"""classifier selection"""
dtc = DecisionTreeClassifier(criterion = 'entropy')
dtc.fit(X_train,y_train)
accuracy = dtc.score(X_test, y_test)
print(accuracy)

prediction = dtc.predict(X_test)
print(prediction)

"""plotting the tree diagram"""
fig,axes = plt.subplots(figsize = (8,10))
plot_tree(dtc, feature_names = X.columns, filled = True, ax = axes, class_names =True, rounded = True)
plt.show()

