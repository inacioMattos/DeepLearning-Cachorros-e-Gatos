import tensorflow as tf
import matplotlib.pyplot as plt
import cv2, os

CATEGORIES = ["Cachorro", "Gato"]

modelPath = r'S:\Machine-Learning\DeepLearning-Cachorros-e-Gatos\models\128-128-128-CNN-noDense.model'
TESTDIR = r"S:\Machine-Learning\DeepLearning-Cachorros-e-Gatos\tests\02"
file = r"S:\Machine-Learning\DeepLearning-Cachorros-e-Gatos\tests\02\5804b25e64d211e18bb812313804a181_7.png"


gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.2)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))



def main():
	model = getModel(modelPath)
	print (predict(model, file, debug=True))
	
	if (file == ""):
		test(model)


def predict(model, img_path, debug=False):
	img = prepare_image(img_path)

	predict = model.predict([img])
	label = int(predict[0][0])

	if (debug):
		print (CATEGORIES[label])
		showImage(img_path)
	return (CATEGORIES[label])


def test(model):
	dog = 0
	cat = 0
	for images in os.listdir(TESTDIR):
		try:
			label = predict(model, os.path.join(TESTDIR, images))

			if (label.lower() == "cachorro"):
				dog += 1
			else:
				cat += 1

			showImage(os.path.join(TESTDIR, images))
		except:
			pass
	
	print ("cachorro = {}\ngatos = {}".format(dog, cat))


def prepare_image(filepath):
	IMG_SIZE = 75
	
	imgArr = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
	imgArr = cv2.resize(imgArr, (IMG_SIZE, IMG_SIZE))
	imgArr = imgArr.reshape(-1, IMG_SIZE, IMG_SIZE, 1)

	return imgArr


def getModel(filepath):
	model = tf.keras.models.load_model(filepath)
	return model


def showImage(img_path):
	img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
	plt.imshow(img)
	plt.show()





main()