import tensorflow as tf
import matplotlib.pyplot as plt
import cv2, os

CATEGORIES = ["Cachorro", "Gato"]

modelPath = r'S:\Machine-Learning\DeepLearning-Cachorros-e-Gatos\models\128-128-128-CNN-noDense.model'
TESTDIR = r"S:\Machine-Learning\DeepLearning-Cachorros-e-Gatos\tests\02"


gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.2)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))




def main():
	model = getModel(modelPath)
	test(model)


def test(model):
	dog = 0
	cat = 0
	for images in os.listdir(TESTDIR):
		try:
			img = prepare_image(os.path.join(TESTDIR, images))
			
			predict = model.predict([img])
			label = int(predict[0][0])
			print (CATEGORIES[label])

			if (label == 0):
				dog += 1
			else:
				cat += 1

			#showImage(os.path.join(TESTDIR, images))
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