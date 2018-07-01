from __future__ import print_function
import keras
import pickle as pk
import numpy as np
import keras.layers as kl
import keras.backend as K
K.set_image_data_format('channels_first')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from keras.utils.vis_utils import plot_model
from Gradient_Reverse_Layer import GradientReversal


class DANN(object):
	def __init__(self, width=28, height=28, channels=3, classes=1, features=1, batch_size=1, grl='auto', summary=False, model_plot=False):
		## Set Defualts
		self.learning_phase = K.variable(1)
		self.domain_invariant_features = None
		self.width, self.height, self.channels = width, height, channels
		self.input_shape = (channels, width, height)
		self.classes = classes
		self.features = features
		self.batch_size = batch_size
		self.grl = 'auto'
		# Set reversal gradient value.
		if grl is 'auto':
			self.grl_rate = 1.0
		else:
			self.grl_rate = grl
		self.summary = summary
		self.model_plot = model_plot

		# Build the model
		self.model = self._build()
		
		# Print and Save the model summary if requested.
		if self.summary:
			self.model.summary()
		if self.model_plot:
			plot_model(self.model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

	def feature_extractor(self, inp):
		''' 
		This function defines the structure of the feature extractor part.
		'''
		out = kl.Conv2D(filters=32, kernel_size=(5, 5), padding="same", activation="relu")(inp)
		out = kl.MaxPooling2D(pool_size=(2, 2))(out)

		out = kl.Conv2D(filters=48, kernel_size=(5, 5), padding="same", activation="relu")(out)
		out = kl.MaxPooling2D(pool_size=(2, 2))(out)

		out = kl.Dropout(0.5)(out)
		out = kl.Flatten()(out)

		feature_output = kl.Dense(self.features, activation="relu")(out)
		self.domain_invariant_features = feature_output
		return feature_output

	def classifier(self, inp):
		''' 
		This function defines the structure of the classifier part.
		'''
		out = kl.Dense(128, activation="relu")(inp)
		out = kl.Dropout(0.5)(out)
		classifier_output = kl.Dense(self.classes, activation="softmax", name="classifier_output")(out)
		return classifier_output

	def discriminator(self, inp):
		''' 
		This function defines the structure of the discriminator part.
		'''
		out = kl.Dense(128, activation="relu")(inp)
		out = kl.Dropout(0.5)(out)
		discriminator_output = kl.Dense(2, activation="softmax", name="discriminator_output")(out)
		return discriminator_output

	def _build(self):
		'''
		This function builds the network based on the Feature Extractor, Classifier and Discriminator parts.
		'''
		inp = kl.Input(shape=self.input_shape, name="main_input")
		feature_output = self.feature_extractor(inp)
		self.grl_layer = GradientReversal(1.0)
		feature_output_grl = self.grl_layer(feature_output)
		labeled_feature_output = kl.Lambda(lambda x: K.switch(K.learning_phase(), K.concatenate([x[:int(self.batch_size//2)], x[:int(self.batch_size//2)]], axis=0), x), output_shape=lambda x: x[0:])(feature_output_grl)

		classifier_output = self.classifier(labeled_feature_output)
		discriminator_output = self.discriminator(feature_output)
		model = keras.models.Model(inputs=inp, outputs=[discriminator_output, classifier_output])
		return model

	def batch_generator(self, trainX, trainY=None, batch_size=1, shuffle=True):
		'''
		This function generates batches for the training purposes.
		'''
		if shuffle:
			index = np.random.randint(0, len(trainX) - batch_size)
		else:
			index = np.arange(0, len(trainX), batch_size)
		while trainX.shape[0] > index + batch_size:
			batch_images = trainX[index : index + batch_size]
			batch_images = batch_images.reshape(batch_size, self.channels, self.width, self.height)
			if trainY is not None:
				batch_labels = trainY[index : index + batch_size]
				yield batch_images, batch_labels
			else:
				yield batch_images
			index += batch_size

	def compile(self, optimizer):
		'''
		This function compiles the model based on the given optimization method and its parameters.
		'''
		self.model.compile(optimizer=optimizer, loss={'classifier_output': 'binary_crossentropy', 'discriminator_output': 'binary_crossentropy'}, loss_weights={'classifier_output': 0.5, 'discriminator_output': 1.0})

	def train(self, trainX, trainDX, trainY=None, epochs=1, batch_size=1, verbose=True, save_model=None):
		'''
		This function trains the model using the input and target data, and saves the model if specified.
		'''
		for cnt in range(epochs):

			# Prepare batch data for the model training.
			Labeled = self.batch_generator(trainX, trainY, batch_size=batch_size // 2)
			UNLabeled = self.batch_generator(trainDX, batch_size=batch_size // 2)
			
			# Settings for learning rate.
			p = np.float(cnt) / epochs
			lr = 0.01 / (1. + 10 * p)**0.75

			# Settings for reverse gradient magnitude (if it's set to be automatically calculated, otherwise set by user.)
			if self.grl is 'auto':
				self.grl_layer.l = 2. / (1. + np.exp(-10. * p)) - 1

			# Re-compile model to adopt new learning rate and gradient reversal value.
			self.compile(keras.optimizers.SGD(lr))

			# Loop over each batch and train the model.
			for batchX, batchY in Labeled:
				# Get the batch for unlabeled data. If the batches are finished, regenerate the batches agian.
				try:
					batchDX = next(UNLabeled)
				except:
					UNLabeled = self.batch_generator(trainDX, batch_size=batch_size // 2)
				# Combine the labeled and unlabeled images along with the discriminative results.
				combined_batchX = np.concatenate((batchX, batchDX))
				batch2Y = np.concatenate((batchY, batchY))
				combined_batchY = np.concatenate((np.tile([0, 1], [batchX.shape[0], 1]), np.tile([1, 0], [batchDX.shape[0], 1])))
				# Train the model
				metrics = self.model.train_on_batch({'main_input': combined_batchX}, {'classifier_output': batch2Y, 'discriminator_output':combined_batchY})
			
			# Print the losses if asked for.
			if verbose:
				print("Epoch {}/{}\n\t[Generator_loss: {:.4}, Discriminator_loss: {:.4}, Classifier_loss: {:.4}]".format(cnt+1, epochs, metrics[0], metrics[1], metrics[2]))
		# Save the model if asked for.
		if save_model is not None and isinstance(save_model, str):
			if save_model[-3:] is not ".h5":
				save_model = ''.join((save_model, ".h5"))
			self.model.save(save_model)
		elif save_model is not None and not isinstance(save_model, str):
			raise TypeError("The input must be a filename for model settings in string format.")


	def evaluate(self, testX, testY=None, weight_loc=None, save_pred=None, verbose=False):
		'''
		This function evaluates the model, and generates the predicted classes.
		'''
		if weight_loc is not None:
			self.compile(keras.optimizers.SGD())
			self.model.load_weights(weight_loc)
		_, yhat_class = self.model.predict(testX, verbose=verbose)
		if save_pred is not None:
			np.save(save_pred, yhat_class)
		if testY is not None and len(testY) == 2:
			acc = self.model.evaluate(testX, testY, verbose=verbose)
			if verbose:
				print("The classifier and discriminator metrics for evaluation are [{}, {}]".format(acc[0], acc[1]))
		elif testY is not None and len(testY) == 1:
			acc = self.model.evaluate(testX, [np.ones((testY.shape[0], 2)), testY], verbose=verbose)
			if verbose:
				print("The classifier metric for evaluation is {}".format(acc[1]))
