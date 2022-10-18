class VGG :
	@staticmethod
		def build(width, height, depth, classes, weightsPath=None):
		# initialize the model
		model = Sequential()
		model.add(Convolution2D(64, (3, 3), activation='relu', padding="same", input_shape=(depth, height, width)))
		#model.add(Convolution2D(64, (3, 3), activation='relu', padding="same"))
		model.add(MaxPooling2D((2, 2), strides=(2, 2), data_format="channels_first"))
        
		model.add(Convolution2D(128, (3, 3), activation='relu', padding="same"))
		#model.add(Convolution2D(128, (3, 3), activation='relu', padding="same"))
		model.add(MaxPooling2D((2, 2), strides=(2, 2), data_format="channels_first"))
        
		model.add(Convolution2D(256, (3, 3), activation='relu', padding="same"))
		#model.add(Convolution2D(256, (3, 3), activation='relu', padding="same"))
		#model.add(Convolution2D(256, (3, 3), activation='relu', padding="same"))
		model.add(MaxPooling2D((2, 2), strides=(2, 2), data_format="channels_first"))
        
		model.add(Convolution2D(512, (3, 3), activation='relu', padding="same"))
		#model.add(Convolution2D(512, (3, 3), activation='relu', padding="same"))
		#model.add(Convolution2D(512, (3, 3), activation='relu', padding="same"))
		model.add(MaxPooling2D((2, 2), strides=(2, 2), data_format="channels_first"))
        
		model.add(Convolution2D(512, (3, 3), activation='relu', padding="same"))
		#model.add(Convolution2D(512, (3, 3), activation='relu', padding="same"))
		#model.add(Convolution2D(512, (3, 3), activation='relu', padding="same"))
		model.add(ZeroPadding2D([1,1]))
		model.add(MaxPooling2D(pool_size=(3, 3), strides=(3, 3), data_format="channels_first"))
        
		model.add(Flatten())
		model.add(Dense(4096, activation='relu'))
		#model.add(Dropout(0.5))
		model.add(Dense(4096, activation='relu'))
		#model.add(Dropout(0.5))
		model.add(Dense(classes, activation='softmax'))
        
		# Si on a deja entrainer le reseau
		# if a weights path is supplied (inicating that the model was
		# pre-trained), then load the weights
		if weightsPath is not None:
			model.load_weights(weightsPath)
            
		# return the constructed network architecture
		return model
	
	
# permet d'enregistrer et de charger les poids d'un modele
ap = argparse.ArgumentParser()
ap.add_argument("-s", "--save-model", type=int, default=-1,help="(optional) whether or not model should be saved to disk")
ap.add_argument("-l", "--load-model", type=int, default=-1,help="(optional) whether or not pretrained model should be loaded")
ap.add_argument("-w", "--weights", type=str, help="(optional) path to weights file")
args = vars(ap.parse_args())


# ---------------------------------------------------------------------------------------
# initialize the optimizer and model
lr=0.1
batch=100
epoch=10
#----------------------------------------------------------------------------------------

print("[INFO] compiling model...")
opt = SGD(lr=lr)
model = VGG.build(width=28, height=28, depth=1, classes=43,weightsPath=args["weights"] if args["load_
model"] > 0 else None)
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

# only train and evaluate the model if we *are not* loading a
# pre-existing model
if args["load_model"] < 0:
    print("[INFO] training...")
    model.fit(trainData, trainLabels, batch_size=batch, epochs=epoch, verbose=1)
    
# show the accuracy on the testing set
print("[INFO] evaluating...")
(loss, accuracy) = model.evaluate(testData, testLabels, batch_size=128, verbose=1)
print("[INFO] accuracy: {:.2f}%".format(accuracy * 100))

# check to see if the model should be saved to file
if args["save_model"] > 0:
    print("[INFO] dumping weights to file...")
    model.save_weights(args["weights"], overwrite=True)