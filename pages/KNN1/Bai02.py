from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

from sklearn import datasets
from skimage import exposure
import numpy as np
import imutils
import cv2
import sklearn


# take the MNIST data and construct the training and testing split, using 75% of the
# data for training and 25% for testing
mnist = datasets.load_digits()
(trainData, testData, trainLabels, testLabels) = train_test_split(np.array(mnist.data),
	mnist.target, test_size=0.25, random_state=42)

# now, let's take 10% of the training data and use that for validation
(trainData, valData, trainLabels, valLabels) = train_test_split(trainData, trainLabels,
	test_size=0.1, random_state=84)

print("training data points: ", len(trainLabels))
print("validation data points: ", len(valLabels))
print("testing data points: ", len(testLabels))

model = KNeighborsClassifier()
model.fit(trainData, trainLabels)
# evaluate the model and update the accuracies list
score = model.score(valData, valLabels)
print("accuracy = %.2f%%" % (score * 100))

# loop over a few random digits
for i in list(map(int, np.random.randint(0, high=len(testLabels), size=(5,)))):
	# grab the image and classify it
	image = testData[i]
	prediction = model.predict(image.reshape(1, -1))[0]

	# convert the image for a 64-dim array to an 8 x 8 image compatible with OpenCV,
	# then resize it to 32 x 32 pixels so we can see it better
	image = image.reshape((8, 8)).astype("uint8")

	image = exposure.rescale_intensity(image, out_range=(0, 255))
	image = imutils.resize(image, width=32, inter=cv2.INTER_CUBIC)

	# show the prediction
	print("I think that digit is: {}".format(prediction))
	cv2.imshow("Image", image)
	cv2.waitKey(0)
