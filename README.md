# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
```
 Step 1.Start
 Step 2.Import the standard Libraries.
 Step 3.Set variables for assigning dataset values.
 Step 4.Import linear regression from sklearn.
 Step 5.Assign the points for representing in the graph.
 Step 6.Predict the regression for marks by using the representation of the graph.
 Step 7.Compare the graphs and hence we obtained the linear regression for the given datas.
 Step 8.End
```
Developed by: KAVIYA SNEKA M
RegisterNumber: 212223040091


```
## Program

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv('student_scores.csv')
print(df)
df.head(0)
df.tail(0)
print(df.head())
print(df.tail())
x = df.iloc[:,:-1].values
print(x)
y = df.iloc[:,1].values
print(y)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)
y_pred = regressor.predict(x_test)
print(y_pred)
print(y_test)
#Graph plot for training data
plt.scatter(x_train,y_train,color='black')
plt.plot(x_train,regressor.predict(x_train),color='blue')
plt.title("Hours vs Scores(Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
#Graph plot for test data
plt.scatter(x_test,y_test,color='black')
plt.plot(x_train,regressor.predict(x_train),color='red')
plt.title("Hours vs Scores(Testing set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
mse=mean_absolute_error(y_test,y_pred)
print('MSE = ',mse)
mae=mean_absolute_error(y_test,y_pred)
print('MAE = ',mae)
rmse=np.sqrt(mse)
print("RMSE= ",rmse)
```

## Output:
Dataset

![image](https://github.com/kaviya546/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/150368823/a9212177-0218-4e21-bc4e-f787826a04df)

Head values

![image](https://github.com/kaviya546/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/150368823/795bddab-6390-498a-a0e6-23259f4eb8a8)

Tail values

![image](https://github.com/kaviya546/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/150368823/5dee723b-5ed0-4f64-b095-4294f1bdc543)

X and Y values

![image](https://github.com/kaviya546/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/150368823/a63fc960-8972-4429-94c4-6ba25dbb5614)

Predication values of X and Y

![image](https://github.com/kaviya546/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/150368823/140971fb-9e5a-4d2d-a848-8327c64247e7)

MSE,MAE and RMSE

![image](https://github.com/kaviya546/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/150368823/bb060ed8-c409-4dd4-a60e-6cc571a397ed)

Training Set

![image](https://github.com/kaviya546/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/150368823/f39f1008-e18f-40f0-ba4e-3bda6f4bb49b)

Testing Set:

![image](https://github.com/kaviya546/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/150368823/08349fde-715d-491c-a3a4-ddbcc53e21bc)



## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
