# Loan Approval Prediction

This project uses machine learning models to predict loan approval status based on applicant information. It employs Decision Tree, Linear Regression, and Logistic Regression classifiers to analyze and predict outcomes.

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Prerequisites](#prerequisites)
- [Project Structure](#project-structure)
- [Features](#features)
- [Usage](#usage)
- [Models Used](#models-used)
- [Results](#results)
- [Visualization](#visualization)
- [License](#license)

## Overview

The primary goal of this project is to predict whether a loan application will be approved or not. This is achieved using three different models:

1. Decision Tree Classifier
2. Linear Regression
3. Logistic Regression

## Dataset

The dataset used is `LoanApprovalPrediction.csv` which includes features like:

- Loan_ID (dropped during preprocessing)
- Gender
- Married
- Education
- Self_Employed
- ApplicantIncome
- CoapplicantIncome
- LoanAmount
- Loan_Amount_Term
- Credit_History
- Property_Area
- Loan_Status (target variable)

## Prerequisites

Ensure you have Python installed and the required libraries before running the project:

```bash
pip install pandas scikit-learn matplotlib
```

## Project Structure

```
LoanApprovalPrediction/
├── LoanApprovalPrediction.csv
└── loan_approval_prediction.py
```

## Features

- Data preprocessing:
  - Handling missing values
  - Encoding categorical variables
  - Feature scaling
- Model training and evaluation:
  - Decision Tree Classifier
  - Linear Regression
  - Logistic Regression
- Accuracy measurement and visualization

## Usage

1. Clone the repository:

```bash
git clone https://github.com/your-repo/LoanApprovalPrediction.git
cd LoanApprovalPrediction
```

2. Ensure your dataset `LoanApprovalPrediction.csv` is in the project directory.

3. Run the script:

```bash
python loan_approval_prediction.py
```

## Models Used

### Decision Tree Classifier
- Entropy criterion for measuring information gain
- Outputs the accuracy and predictions
- Plots the decision tree

### Linear Regression
- Scaled data input
- Outputs accuracy and predictions
- Visualizes the relationship between LoanAmount and predictions

### Logistic Regression
- Outputs accuracy and predictions
- Visualizes the classifier output

## Results

- Decision Tree Classifier Accuracy: ~[Dynamic Output]%
- Linear Regression Accuracy: ~[Dynamic Output]%
- Logistic Regression Accuracy: ~[Dynamic Output]%

## Visualization

The project includes the following visualizations:

1. Decision Tree structure
2. Linear Regression scatter plot
3. Logistic Regression plot
