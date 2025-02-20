"""Importing required libraries and functions"""
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import style
from sklearn.tree import DecisionTreeClassifier, plot_tree, DecisionTreeRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_validate
from sklearn import preprocessing

style.use('fivethirtyeight')

data = pd.read_csv('LoanApprovalPrediction.csv')
#print(data.head(10))

"""inspecting the data"""
#data.info()
"""checking for columns with null values"""
#print(data.isnull().sum())

"""dropping unnecessary columns"""
data = data.drop('Loan_ID', axis =1)
#data.info()

"""encoding non-categorical values"""

le = LabelEncoder()


data['Education'] = le.fit_transform(data['Education'])
data['Married'] = le.fit_transform(data['Married'])
data['Self_Employed'] = le.fit_transform(data['Self_Employed'])
data['Gender'] = le.fit_transform(data['Gender'])
data['Property_Area'] = le.fit_transform(data['Property_Area'])
data['Loan_Status'] = le.fit_transform(data['Loan_Status'])


"""making all int32 data types int64"""
for col in data.columns:
    if data[col].dtype == 'int32':
        data[col] = data[col].astype('int64')

        
"""dealing with missing values or NaNs"""

data['Dependents'] = data['Dependents'].fillna(data['Dependents'].mode().iloc[0])
data['LoanAmount'] = data['LoanAmount'].fillna(data['LoanAmount'].mean())
data['Loan_Amount_Term'] = data['Loan_Amount_Term'].fillna(data['Loan_Amount_Term'].mode().iloc[0])
data['Credit_History'] = data['Credit_History'].fillna(data['Credit_History'].mode().iloc[0])

"""feature selection"""
X = data.drop('Loan_Status', axis=1)
X_new = preprocessing.scale(X)
#print(X['Married'].dtype)
y = data['Loan_Status']
"""checking the structure of our y variable"""
#print(y.unique())

"""training and testing"""
X_train, X_test, y_train,  y_test = train_test_split(X, y , test_size = 0.3)

clf = DecisionTreeClassifier(criterion = "entropy")#to measure the level of uncertainty in every stage.

clf.fit(X_train, y_train)

accuracy = clf.score(X_test,y_test)
print(f'Accuracy of DTC: {accuracy*100}')

predictions = clf.predict(X_test)
print(f'DTC predicted values: {predictions}')

"""plotting the decision tree diagram"""
fig, axes = plt.subplots(figsize=(20,50))
plot_tree(clf, filled = True, feature_names = X.columns, class_names = True, rounded = True, ax = axes)
plt.show()

#X_train, X_test, y_train,  y_test = train_test_split(X_new, y , test_size = 0.3) for logistic and linear regression models.
"""using a linearregression classifier, with scaled data"""
my_clf = LinearRegression()
my_clf.fit(X_train,y_train)

reg_accuracy = my_clf.score(X_test,y_test)
print(f'Linear Refression Accuracy: {reg_accuracy*100}')

rg_predict = my_clf.predict(X_test)
print(f'Linear Regression predicted values: {rg_predict}')
"""plotting the linear regression"""
plt.scatter(X_test['LoanAmount'], y_test)
plt.plot(rg_predict)
plt.show()

##logistic regression.
"""logistic regression"""
log_clf = LogisticRegression()
log_clf.fit(X_train, y_train)

log_accuracy = log_clf.score(X_test, y_test)
print(f'Logistic classifier accuracy score: {log_accuracy*100}')

log_predict = log_clf.predict(X_test)
print(f'Logistic Regressor predcited Values: {log_predict}')

"""plotting the logistic regressor"""
plt.scatter(X[:,8], y, c=y)
plt.show()

