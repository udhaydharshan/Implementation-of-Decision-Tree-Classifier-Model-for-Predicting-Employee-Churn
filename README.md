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
data=pd.read_csv("C:/Users/acer/Downloads/Employee.csv")
print("data.head():")
data.head()
```
```
print("data.info():")
data.info()
```
```
print("isnull() and sum():")
data.isnull().sum()
```
```
print("data value counts():")
data["left"].value_counts()
```
```
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
```
```
print("data.head() for Salary:")
data["salary"]=le.fit_transform(data["salary"])
data.head()
```
```
print("x.head():")
x=data[["satisfaction_level","last_evaluation","number_project","average_montly_hours","time_spend_company","Work_accident","promotion_last_5years","salary"]]
x.head()
```
```
y=data["left"]
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=100)
from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(criterion="entropy")
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)
```
```
print("Accuracy value:")
from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy
```
```
print("Data Prediction:")
dt.predict([[0.5,0.8,9,260,6,0,1,2]])
```
```
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

plt.figure(figsize=(8,6))
plot_tree(dt, feature_names=x.columns, class_names=['salary', 'left'], filled=True)
plt.show()
```

## Output:
<img width="992" height="198" alt="Screenshot 2026-02-25 093941" src="https://github.com/user-attachments/assets/bbe7ea13-aa84-4358-af5b-96094757b82e" />
<img width="917" height="286" alt="Screenshot 2026-02-25 094005" src="https://github.com/user-attachments/assets/ebcdb9c8-cda6-45fb-8258-e1b79be4df51" />
<img width="798" height="208" alt="Screenshot 2026-02-25 094015" src="https://github.com/user-attachments/assets/39225d9b-9dd3-40fe-a84c-d181a69d1cc8" />
<img width="980" height="95" alt="Screenshot 2026-02-25 094030" src="https://github.com/user-attachments/assets/dba3998a-b3f0-4255-a95a-02f2ed056222" />
<img width="1045" height="216" alt="Screenshot 2026-02-25 094042" src="https://github.com/user-attachments/assets/156d4c46-025f-4ac7-ad8d-a3dabdc9b59f" />
<img width="1006" height="163" alt="Screenshot 2026-02-25 094058" src="https://github.com/user-attachments/assets/2c630497-5ae7-40c3-b596-380a9269b10b" />
<img width="980" height="58" alt="Screenshot 2026-02-25 094107" src="https://github.com/user-attachments/assets/0dc3cf60-765e-4352-b416-99bfc33bc570" />
<img width="1006" height="122" alt="Screenshot 2026-02-25 094118" src="https://github.com/user-attachments/assets/ae0ccb15-21f9-4fee-ac0a-7a4fdcbaa0b4" />




## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
