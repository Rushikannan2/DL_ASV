import tensorflow as tf
from feature_extraction import load_features_from_h5
from cnn_model import create_cnn_model  # Import the CNN creation function

def train_model():
    # Load preprocessed features
    X_train, y_train = load_features_from_h5('processed_features.h5')

    # Reshape the features to match the input shape expected by the CNN
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], X_train.shape[2], 1))

    # Split the data into training and validation sets (dev data)
    X_train, X_dev = X_train[:int(0.8*len(X_train))], X_train[int(0.8*len(X_train)):]
    y_train, y_dev = y_train[:int(0.8*len(y_train))], y_train[int(0.8*len(y_train)):]

    # Create and train the model
    model = create_cnn_model((X_train.shape[1], X_train.shape[2], 1))
    model.fit(X_train, y_train, epochs=10, validation_data=(X_dev, y_dev))

    # Save the trained model in the Keras format
    model.save('baseline_model.keras')  # Changed to Keras format
if __name__ == '__main__':
    train_model()
