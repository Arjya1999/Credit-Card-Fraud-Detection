import pandas as pd
import numpy as np
import keras
from sklearn.metrics import classification_report
dataset=pd.read_csv('creditcard.csv')

#Data Preprocessing 
from sklearn.preprocessing import StandardScaler
sc_time = StandardScaler()
dataset["Time"] = sc_time.fit_transform(dataset["Time"].values.reshape(-1,1))
dataset["Amount"] = sc_time.fit_transform(dataset["Amount"].values.reshape(-1,1))
dataset=dataset.drop(columns=['V23','V24'])

#Cerating Dependent and Independent Features
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

from sklearn.ensemble import IsolationForest
classifier=IsolationForest(n_estimators=100, max_samples=len(X),contamination=0.00173,random_state=0)
classifier.fit(X_train)
y_pred = classifier.predict(X_test)
y_pred[y_pred == 1] = 0
y_pred[y_pred == -1] = 1

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

print("Classification Report :")
print(classification_report(y_test,y_pred))
