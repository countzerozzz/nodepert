
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
from tensorflow.keras import datasets, layers, models

# import data_loaders.mnist_loader as data
# import data_loaders.cifar10_loader as data
import data_loaders.cifar100_loader as data
# import data_loaders.tiny_imagenet_loader as data

print("GPU Available: ", tf.config.list_physical_devices('GPU'))

num_examples = data.dataset_info.splits['train'].num_examples
num_classes = data.num_classes

batchsize = 128
height, width, channels = data.dataset_info.features['image'].shape
train_ds = data.get_rawdata_batches(batchsize=batchsize, split=data.trainsplit, return_tfdata=True)
test_ds = data.get_rawdata_batches(batchsize=batchsize, split=data.testsplit, return_tfdata=True)

def normalize_img(image, label):
  """Normalizes images: `uint8` -> `float32`."""
  return tf.cast(image, tf.float32) / 255., label

train_ds = train_ds.map(
    normalize_img, num_parallel_calls=tf.data.AUTOTUNE)

train_ds = train_ds.shuffle(num_examples)

test_ds = test_ds.map(
    normalize_img, num_parallel_calls=tf.data.AUTOTUNE)

model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(height, width, channels)),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dense(10)
])

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(height, width, channels)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
# model.add(layers.Dense(num_classes), activation='sigmoid')
model.add(layers.Dense(num_classes))

model.summary()
model.compile(
    optimizer=tf.keras.optimizers.Adam(0.001),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
)


model.fit(
    train_ds,
    epochs=6,
    validation_data=test_ds,
)
