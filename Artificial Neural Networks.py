import pandas as pd
import numpy as np
import keras
dataset=pd.read_csv('creditcard.csv')

#Data Preprocessing 
from sklearn.preprocessing import StandardScaler
sc_time = StandardScaler()
dataset["Time"] = sc_time.fit_transform(dataset["Time"].values.reshape(-1,1))
dataset["Amount"] = sc_time.fit_transform(dataset["Amount"].values.reshape(-1,1))

#Cerating Dependent and Independent Features
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout

# Initialising the ANN
classifier = Sequential()
# Adding the input layer and the first hidden layer
classifier.add(Dense(units = 16, kernel_initializer = 'uniform', activation = 'relu', input_dim = 30))
classifier.add(Dropout(p = 0.2))

# Adding the second hidden layer
classifier.add(Dense(units = 16, kernel_initializer = 'uniform', activation = 'relu'))
classifier.add(Dropout(p = 0.2))

# Adding the output layer
classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))

# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting the ANN to the Training set
classifier.fit(X_train, y_train, batch_size = 10, epochs = 100)

y_pred=classifier.predict(X_test)
y_pred = (y_pred > 0.5)

score=classifier.evaluate(X_test,y_test)
print(score)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
