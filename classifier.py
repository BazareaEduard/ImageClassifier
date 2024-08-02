import tensorflow as tf
import cv2, imghdr
import os
from matplotlib import pyplot as plt 
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
from keras.metrics import Precision, Recall, CategoricalAccuracy
from keras.callbacks import EarlyStopping
import numpy as np


# GPU setup
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus: 
    tf.config.experimental.set_memory_growth(gpu, True)
tf.config.list_logical_devices('GPU')

# Image manipulation
allowed_formats = ['jpeg', 'jpg', 'png', 'bmp']
base_folder = "data"
size_threshold_kb = 10

for folder in os.listdir(base_folder):
    folder_path = os.path.join(base_folder, folder)
    if os.path.isdir(folder_path):
        for file in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file)
            file_ext = file.split('.')[-1].lower()
            file_size_kb = os.path.getsize(file_path) / 1024
            try:
                img = cv2.imread(file_path)
                tip = imghdr.what(file_path)
                if file_size_kb < size_threshold_kb:
                    print(f'File {file_path} is under {size_threshold_kb} KB')
                    os.remove(file_path)
                if tip not in allowed_formats:
                    print(f'Image not in ext list {file_path} (detected as {tip})')
                    os.remove(file_path)
                elif img is None:
                    print(f'Image is corrupted {file_path}')
                    os.remove(file_path)
            except Exception as e:
                print(f'Issue with image {file_path}: {e}')

# Data loader
data = tf.keras.utils.image_dataset_from_directory('data', batch_size=32, image_size=(256, 256))

# Get the number of batches per epoch
total_samples = sum(len(files) for _, _, files in os.walk('data') if files)
batch_size = 32
steps_per_epoch = total_samples // batch_size

print(f'Total samples: {total_samples}')
print(f'Batch size: {batch_size}')
print(f'Steps per epoch: {steps_per_epoch}')

# Scale data
data = data.map(lambda x, y: (x / 255.0, y))

# Split data
dataset_size = len(data)
train_size = int(0.7 * dataset_size)
val_size = int(0.2 * dataset_size)
test_size = int(0.1 * dataset_size)

train = data.take(train_size)
val = data.skip(train_size).take(val_size)
test = data.skip(train_size + val_size).take(test_size)

print(f'Train size: {train_size}')
print(f'Validation size: {val_size}')
print(f'Test size: {test_size}')

# Model
model = Sequential([
    Conv2D(16, (3, 3), activation='relu', input_shape=(256, 256, 3)),
    MaxPooling2D(),
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D(),
    Conv2D(16, (3, 3), activation='relu'),
    MaxPooling2D(),
    Flatten(),
    Dense(256, activation='relu'),
    Dense(6, activation='softmax')  # 6 classes
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()

# Early stopping callback
early_stopping = EarlyStopping(monitor = 'accuracy', patience = 2, restore_best_weights = True)

# Train
hist = model.fit(train, epochs=20, validation_data=val, callbacks=[early_stopping])

# Plot loss
fig = plt.figure()
plt.plot(hist.history['loss'], color='teal', label='loss')
plt.plot(hist.history['val_loss'], color='orange', label='val_loss')
fig.suptitle('Loss', fontsize=20)
plt.legend(loc="upper left")
plt.show()

# Plot accuracy
fig = plt.figure()
plt.plot(hist.history['accuracy'], color='teal', label='accuracy')
plt.plot(hist.history['val_accuracy'], color='orange', label='val_accuracy')
fig.suptitle('Accuracy', fontsize=20)
plt.legend(loc="upper left")
plt.show()

# Evaluate
pre = Precision()
re = Recall()
acc = CategoricalAccuracy()

for batch in test.as_numpy_iterator(): 
    X, y = batch
    yhat = model.predict(X)
    yhat = np.argmax(yhat, axis=1)
    pre.update_state(y, yhat)
    re.update_state(y, yhat)
    acc.update_state(y, yhat)

print(f'Precision: {pre.result().numpy()}, Recall: {re.result().numpy()}, Accuracy: {acc.result().numpy()}')

# Test
img = cv2.imread('images_to_use/sign1.png')
plt.imshow(img)
plt.show()
resize = tf.image.resize(img, (256, 256))
plt.imshow(resize.numpy().astype(int))
plt.show()
yhat = model.predict(np.expand_dims(resize / 255.0, 0))
print(yhat)
predicted_class = np.argmax(yhat)
print(f'Predicted class is {predicted_class}')

# Save model
model.save(os.path.join('models','imageclassifier.h5'))