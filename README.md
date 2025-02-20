# GIT-PROJECT
Project 1: This python code provides a comprehensive code into how we can utilize machine learning to 
predict stock prices. However this code is a dummy code. This is because no actual stock prices were 
analyzed. However, this code displays efficient use of machine learning techniques such as cross validation, linear regression
scikit-learn and some useful statistics. In due course actual stock price data will be used specifically from S&P 500 stock data from 
3 decades ago.

Project 2: DECISION TREE PROJECT
This python script produces a decision tree based on a machine learning algorithm
to decide whether an individual will proceed to buy a computer based on certain factors
These factors include: their age, whether they are a student or not, their income and their 
credit rating/history. 
I first begun by importing the necessary libraries required for machine learning. these include,
scikit-learn; a library that is to some a extent a father to all machine learning modules. Out of Scikit-Learn
or sklearn, I imported necessary functions such as DecisionTreeClassifiers, preprocessing, train-test-split,
metrics fot accuracy-score, etc.
Other really important libraries include matplotlib fot data visualization, pandas for data retrieval and dataframe
creation, labelencoders for encoding or converting our string data to digits, etc.
the next step was to do a data cleaning which basically involves cleaning the data and ridding it of outliers, NaN values/null values
and dropping unncecessary columns. 
Now I proceed to scaling the data by standardizing it.
Finally, I trained the model with 80% of its data reserving 20% for testing
Using the DecisionTreeClassifier as "dtc" variable.
I then measured the accuracy of the test data and predicted new values based on the historical data.
The last stage was a plot of out decision tree showcasing the stages in a tree-like manner. 
