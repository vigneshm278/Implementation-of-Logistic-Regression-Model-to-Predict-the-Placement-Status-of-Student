# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Data Preparation
2. Feature Scaling
3. Train-Test Split
4. Train the Logistic Regression Model 

## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: VIGNESH M
RegisterNumber: 212223240176
*/


import pandas as pd
data=pd.read_csv('Placement_Data.csv')
data.head()
```
![image](https://github.com/user-attachments/assets/edb448ca-c2b6-4a2b-83df-b33f5e2af876)

```
data1=data.copy()
data1=data1.drop(["sl_no","salary"],axis=1)
data1.head()
```

![image](https://github.com/user-attachments/assets/34f8eba1-44cf-481b-b6f0-e86b7d671f92)
```

data1.isnull().sum()

```

![image](https://github.com/user-attachments/assets/f753433b-8ea8-4676-acb5-307c04b7a714)
```

data1.duplicated().sum()
```

![image](https://github.com/user-attachments/assets/6e787434-1a72-4b32-bc5c-6b0c03848dbf)

```
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data1["gender"]=le.fit_transform(data1["gender"])
data1["ssc_b"]=le.fit_transform(data1["ssc_b"])
data1["hsc_b"]=le.fit_transform(data1["hsc_b"])
data1["hsc_s"]=le.fit_transform(data1["hsc_s"])
data1["degree_t"]=le.fit_transform(data1["degree_t"])
data1["workex"]=le.fit_transform(data1["workex"])
data1["specialisation"]=le.fit_transform(data1["specialisation"])
data1["status"]=le.fit_transform(data1["status"])
data1

```
![image](https://github.com/user-attachments/assets/9c27dabf-e140-4aae-a03f-6dfeb4db46b3)

```
x=data1.iloc[:, : -1]
x
```
![image](https://github.com/user-attachments/assets/f5eb1a47-faa9-4faa-b9a7-e970b7d5f466)
```

y=data1["status"]
y
```
![image](https://github.com/user-attachments/assets/c1a90363-2667-44cc-8dc1-679b798571a9)

```
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(solver="liblinear")
lr.fit(x_train,y_train)
y_pred=lr.predict(x_test)
y_pred
```
![image](https://github.com/user-attachments/assets/b6584007-6ef9-4f44-b66f-8cfe86986a6c)
```
from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,y_pred)
accuracy
```
![image](https://github.com/user-attachments/assets/f99b136a-1a77-450e-9d64-e7d6dc0dacbb)
```
from sklearn.metrics import confusion_matrix
confusion=confusion_matrix(y_test,y_pred)
confusion
```
![image](https://github.com/user-attachments/assets/438d4c68-93ad-452b-8ef5-48ee17b90266)c

## Output:



## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
