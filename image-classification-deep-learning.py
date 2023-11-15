import numpy as np
import pickle
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, BatchNormalization, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import plot_model
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping
from keras.preprocessing import image


# Function to load and unpickle the CIFAR-10 dataset
def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

# Function to load all CIFAR-10 data
def load_cifar10_data(data_dir):
    train_data = None
    train_labels = []

    # Load all train batches
    for i in range(1, 6):
        data_batch_i = unpickle(f"{data_dir}/data_batch_{i}")
        if i == 1:
            train_data = data_batch_i[b'data']
        else:
            train_data = np.vstack((train_data, data_batch_i[b'data']))
        train_labels += data_batch_i[b'labels']

    # Load test batch
    test_data_dict = unpickle(f"{data_dir}/test_batch")
    test_data = test_data_dict[b'data']
    test_labels = test_data_dict[b'labels']

    train_data = train_data.reshape((len(train_data), 3, 32, 32)).transpose(0, 2, 3, 1)
    test_data = test_data.reshape((len(test_data), 3, 32, 32)).transpose(0, 2, 3, 1)
    
    # Normalize data
    train_data = train_data.astype('float32') / 255.0
    test_data = test_data.astype('float32') / 255.0

    # Convert class vectors to binary class matrices (one-hot encoding)
    train_labels = to_categorical(train_labels, 10)
    test_labels = to_categorical(test_labels, 10)

    return (train_data, train_labels), (test_data, test_labels)

# Load the data
data_dir = './cifar-10-batches-py'
(train_images, train_labels), (test_images, test_labels) = load_cifar10_data(data_dir)

# CNN architecture
model = Sequential([
    Conv2D(32, (3, 3), padding='same', input_shape=train_images.shape[1:], activation='relu'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),
    
    Conv2D(64, (3, 3), padding='same', activation='relu'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),
    
    Conv2D(128, (3, 3), padding='same', activation='relu'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),

    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(10, activation='softmax')
])

# Compile the model
model.compile(loss='categorical_crossentropy',
              optimizer=Adam(lr=0.001),
              metrics=['accuracy'])

plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

# Early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=3, verbose=1, mode='min', restore_best_weights=True)

# Train the model
history = model.fit(train_images, train_labels,
                    batch_size=64,
                    epochs=20,
                    validation_data=(test_images, test_labels),
                    shuffle=True,
                    callbacks=[early_stopping])

# Plot the training and validation accuracy and loss
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy over epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss over epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Evaluate the model on test data
test_loss, test_acc = model.evaluate(test_images, test_labels)
print('Test accuracy:', test_acc)

# Display a few images from the test set and predict their labels
def display_test_samples_and_predictions(model, test_images, test_labels, num_samples=5):
    indices = np.random.choice(range(len(test_images)), num_samples)
    for i, index in enumerate(indices):
        img = test_images[index]
        plt.imshow(img)
        plt.show()
        actual_label = np.argmax(test_labels[index])
        pred_label = np.argmax(model.predict(img.reshape(1, 32, 32, 3)))
        print(f"Sample {i+1}: Actual Label = {actual_label}, Predicted Label = {pred_label}")

display_test_samples_and_predictions(model, test_images, test_labels)