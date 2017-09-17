from keras.datasets import mnist
from keras.models import Sequential,load_model
from keras.layers  import Dense,Dropout,Conv2D,MaxPooling2D,Flatten
from keras.utils import to_categorical
import os
#Split the datasets into training and testing modules

(X_train,Y_train),(X_test,Y_test)=mnist.load_data()
X_train.resize(60000,28,28,1)
X_test.resize(10000,28,28,1)
Y_train=to_categorical(Y_train,num_classes=10)
Y_test=to_categorical(Y_test,num_classes=10)
#Create a Convolutional Neural Network
if not os.path.exists('conv_model1.h5'):
	model=Sequential()
	model.add(Conv2D(filters=20,kernel_size=(3,3),padding='same',activation='relu',input_shape=X_train.shape[1:]))
	model.add(MaxPooling2D(pool_size=(2,2)))
	model.add(Conv2D(filters=32,kernel_size=(3,3),padding='same',activation='relu'))
	model.add(MaxPooling2D(pool_size=(2,2)))
	model.add(Dropout(0.4))
	#model.add(Conv2D(filters=32,kernel_size=(3,3),padding='same',activation='relu'))
	#model.add(MaxPooling2D(pool_size=(2,2)))

	## Create the last Dense block

	model.add(Flatten())
	model.add(Dense(80,activation='relu'))
	#model.add(Dense(80,activation='relu'))
	model.add(Dropout(0.4))
	#model.add(Dense(80,activation='relu'))
	model.add(Dense(10,activation='softmax'))

	model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

	##Start training

	model.fit(X_train[:1000],Y_train[:1000],epochs=100)
	model.save('conv_model1.h5')
else:
	model=load_model('conv_model1.h5')
	print('\n')
	print('Accuracy :',model.evaluate(X_test,Y_test)[1]*100,"%")
