import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import TensorBoard
import pickle, os, time


DATADIR="data/"
NAME="cachorros-gatos-cnn-64x3-{}".format(int(time.time()))

tensorboard = TensorBoard(log_dir="logs/{}".format(NAME))


def getData():
	X = pickle.load(open(DATADIR + "X.pickle", "rb"))
	y = pickle.load(open(DATADIR + "y.pickle", "rb"))

	return X, y


def normalizeData(X):
	return X/255.0	# já que numa imagem o valor máximo é 255 para cada pixels, é só dividir por 255.


def saveModel(model):
	model.save("64*2x3-CNN.model")


def trainModel(model, training_set):
	X, y = training_set

	model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
	model.fit(X, y, batch_size=32, validation_split=0.1, epochs=7, callbacks=[tensorboard])
	return model


def createModel(X):
	model = Sequential()

	model.add(Conv2D(64, (3,3), input_shape=X.shape[1:]))
	model.add(Activation("relu"))
	model.add(MaxPooling2D(pool_size=(2, 2)))

	model.add(Conv2D(128, (4,4)))
	model.add(Activation("relu"))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	
	model.add(Conv2D(256, (3,3)))
	model.add(Activation("relu"))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	
	model.add(Flatten())

	model.add(Dense(1))
	model.add(Activation("sigmoid"))

	return model


def main():
	X, y = getData()
	X = normalizeData(X)
	model = createModel(X)
	model = trainModel(model, (X, y))
	saveModel(model)



main()