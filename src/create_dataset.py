import numpy as np
import matplotlib.pyplot as plt
import os, cv2, random, pickle

DATADIR = "S:\Machine-Learning\DeepLearning-Cachorros-e-Gatos\data"
CATEGORIES = ["Dog", "Cat"]
IMG_SIZE = 75


def main():
	dataset = create_training_data()
	print (len(dataset))

	X, y = normalizeData(dataset)
	saveData(X, y)


def showImage(img):
	plt.imshow(img, cmap="gray")
	plt.show()


def saveData(X, y):
	pickle_out = open("X.pickle", "wb")
	pickle.dump(X, pickle_out)
	pickle_out.close()

	pickle_out = open("y.pickle", "wb")
	pickle.dump(y, pickle_out)
	pickle_out.close()


def normalizeData(data):
	X = []
	y = []

	for features, label in data:
		X.append(features)
		y.append(label)

	X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1)

	return X, y


def create_training_data():
	training_data = []

	for categoria in CATEGORIES:
		basePath = os.path.join(DATADIR, categoria)
		
		for img in os.listdir(basePath):
			
			try:
				currentPath = os.path.join(basePath, img)
				imgArr = cv2.imread(currentPath, cv2.IMREAD_GRAYSCALE)
				# se quisesse colorida "cv2.cvtColor(cv2.imread(currentPath), cv2.COLOR_BGR2RGB)"
				imgArr = cv2.resize(imgArr, (IMG_SIZE, IMG_SIZE))

				training_data.append([imgArr, CATEGORIES.index(categoria)])
			
			except Exception as err:
				pass

	return training_data
			



		

main()