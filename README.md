# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import required libraries

2.Load the employee dataset

3.Preprocess the data (handle missing values & encode categorical data)

4.Split the dataset into training and testing sets

5.Train the Decision Tree Classifier

6.Predict employee churn on test data

7.Evaluate the model using accuracy

## Program:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: UDHAYDHARSHAN S
RegisterNumber: 212225230286 
*/
```
```
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

data = pd.read_csv("C:/Users/acer/Downloads/Employee.csv")

data.rename(columns={'Departments ': 'Departments'}, inplace=True)

le = LabelEncoder()
for column in data.columns:
    if data[column].dtype == 'object':
        data[column] = le.fit_transform(data[column])

X = data.drop('left', axis=1)   
y = data['left']                

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = DecisionTreeClassifier(criterion='entropy', random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy of Decision Tree Classifier:", accuracy)
```

## Output:
<img width="412" height="24" alt="image" src="https://github.com/user-attachments/assets/b4b0242d-619f-4ae0-9ffe-53939c8cdde5" />



## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
