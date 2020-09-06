import tensorflow as tf
from tensorflow.keras import Sequential, layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator

dataset_dir = 'dataset'
batch_size = 8
img_height = 100
img_width = 100

data_generator = ImageDataGenerator(
    rescale=1.0/255,
    validation_split=0.2,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

training_data_generator = data_generator.flow_from_directory(
    dataset_dir,
    target_size=(img_height, img_width),
    subset="training",
    batch_size=batch_size,
)

test_data_generator = data_generator.flow_from_directory(
    dataset_dir,
    target_size=(img_height, img_width),
    subset="validation",
    batch_size=batch_size,
)

model = Sequential()

# Write your own code
# Add your own layers here!



model.compile(loss='categorical_crossentropy',
    optimizer='rmsprop', # Try a different optimizer!
    metrics=['accuracy']
)

model.summary()

model.fit(
    training_data_generator,
    validation_data=test_data_generator,
    epochs=3)
