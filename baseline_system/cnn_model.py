# cnn_model.py

from tensorflow.keras import layers, models

def create_cnn_model(input_shape=(159, 13, 1)):
    """
    Create and return a CNN model for audio classification.
    """
    model = models.Sequential()

    # Convolutional layer
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(layers.MaxPooling2D((2, 2)))

    # Another convolutional layer
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))

    # Flatten and fully connected layer
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))

    # Output layer for binary classification (bonafide or spoof)
    model.add(layers.Dense(1, activation='sigmoid'))

    # Compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    return model
