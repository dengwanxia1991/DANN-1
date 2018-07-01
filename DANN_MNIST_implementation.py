import keras
import numpy as np
import pickle as pk
from dann_helper import DANN

# Import the MNIST dataset with labels
(trainX, trainY), (_, _) = keras.datasets.mnist.load_data()

# Process data to match the model specs.
trainX = np.expand_dims(trainX, axis=3)
trainX = np.tile(trainX, [1, 1, 1, 3]).transpose(0, 3, 1, 2)

# Divide train and test sets (the only limitation of this model is to assign even number as batch_sizes and the size of data should yield 0 remainder.)
trainX, testX = trainX[:45000], trainX[45000:55016]
trainY = keras.utils.to_categorical(trainY, num_classes=10)
trainY, testY = trainY[:45000], trainY[45000:55016]

# Load MNIST-M dataset as out-of-domain and unlabeled data
with open('./mnistm_data.pkl', 'rb') as f:
	mnist_m = pk.load(f)		
trainDX = mnist_m['train'].transpose(0, 3, 1, 2)[:45000]
testDX = mnist_m['train'].transpose(0, 3, 1, 2)[45000:55016]
mnist_m = None

# Rescale -1 to 1
trainX = (trainX.astype(np.float32) - 127.5) / 127.5
trainDX = (trainDX.astype(np.float32) - 127.5) / 127.5
testX = (testX.astype(np.float32) - 127.5) / 127.5
testDX = (testDX.astype(np.float32) - 127.5) / 127.5

# Initiate the model
dann = DANN(summary=True, width=28, height=28, channels=3, classes=10, features=32, batch_size=32, model_plot=True)
# Train the model
dann.train(trainX, trainDX, trainY, epochs=100, batch_size=32, plot_intervals=20)
# Evaluate for binary MNIST
dann.evaluate(testX, testY, save_pred="./binary_testX.npy", verbose=True)
# Evaluate for colorful MNIST (MNIST-M)
dann.evaluate(testDX, save_pred="./colorful_testX.npy", verbose=True)
