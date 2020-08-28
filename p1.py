from tensorflow import keras
from tensorflow.keras import layers, models

# Load the data
(training_set, training_labels), (test_set, test_labels) = keras.datasets.mnist.load_data()
training_set = keras.utils.normalize(training_set, axis=1)
test_set = keras.utils.normalize(test_set, axis=1)

# Create a new model
model = models.Sequential()

# Add layers to the model
# Write your own code here!

# Compile the model
model.compile(	optimizer='adam',
   	loss='sparse_categorical_crossentropy',
    	metrics=['accuracy'] 	   )

# Print model summary
model.summary()

# Train the model
model.fit(training_set, training_labels, epochs=3)

# Validate the model
print('[val_loss, val_acc] =', model.evaluate(test_set, test_labels))
