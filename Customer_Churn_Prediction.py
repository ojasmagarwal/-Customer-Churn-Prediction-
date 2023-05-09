#PART 1 - DATA PREPROCESSING

#importing the modules

import pandas as pd
import numpy as np
import tensorflow as tf

#importing the dataset

dataset = pd.read_csv('Churn_Modelling.csv')
x = dataset.iloc[:,3:-1].values
y = dataset.iloc[:,-1].values 

# Encoding the Dependent Variable
from sklearn.preprocessing import LabelEncoder  #encoding gender
le = LabelEncoder()
x[:,2] = le.fit_transform(x[:,2])

# Encoding categorical data 
from sklearn.compose import ColumnTransformer  #encoding countires
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [1])], remainder='passthrough')
x = np.array(ct.fit_transform(x))

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 1)

# Feature Scaling  #it is a must for DL
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

#PART 2 - BUILDING ANN

#Initializng the ANN

ann = tf.keras.models.Sequential() #sequence of layers by tensorflow

#Adding the input layer and the first hidden layer

ann.add(tf.keras.layers.Dense(units=6,activation='relu')) #we choose the no of neurons by ourselves and we take a random number and by experimenting we know which no gives the max accuracy
#activation is the activation function and relu means rectifier function

#Adding the second hidden layer

ann.add(tf.keras.layers.Dense(units=6,activation='relu'))

#Adding the output layer

ann.add(tf.keras.layers.Dense(units=1,activation='sigmoid')) #the units parametere here is 1 because in this dataset we have binary output so 1

#PART 3 -TRAINING THE ANN

#Compiling the ANN

ann.compile(optimizer = 'adam' ,loss = 'binary_crossentropy' ,metrics = ['accuracy']) 
#since we have binary outcome we will put binary_Crossentropy in loss but if we have 3 classes we will put 'categorical_crossentropy
#optimiser is the stochastic gradient descent coded as adam

# Training the ANN on the training set

ann.fit(x_train,y_train,batch_size= 32,epochs=100) #batch size gives the no of predictions we want in the batch to be compared to that same no of real results (generally 32)

# Predicting the Test set results
y_pred = ann.predict(x_test)
y_pred = (y_pred > 0.5)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print(cm)
score = accuracy_score(y_test, y_pred) 
print(score)

