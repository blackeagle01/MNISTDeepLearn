from keras.datasets import mnist
from keras.layers import Dense,Dropout
from keras.models import Sequential,load_model
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import train_test_split
from keras.utils import to_categorical
import numpy as np
import os
#Load training data

(X_train,Y_train),(X_test,Y_test)=mnist.load_data()
X_train.resize(60000,784)
X_test.resize(10000,784)
X_train=StandardScaler().fit_transform(X_train)
X_test=StandardScaler().fit_transform(X_test)
Y_train=to_categorical(Y_train,10)
Y_test=to_categorical(Y_test,10)

if not os.path.exists('myperceptron.h5'):
	#Build model

	model=Sequential()
	model.add(Dense(900,activation='relu',input_dim=784))
	model.add(Dense(900,activation='relu'))
	model.add(Dropout(0.5))
	model.add(Dense(10,activation='softmax'))
	model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

	#Train Model

	model.fit(X_train,Y_train,epochs=30)
	model.save('myperceptron.h5')
else:
	model=load_model('myperceptron.h5')
	print(model.evaluate(X_test,Y_test))