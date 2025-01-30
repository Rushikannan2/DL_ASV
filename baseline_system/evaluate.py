import tensorflow as tf
import h5py
from feature_extraction import load_features_from_h5

def evaluate_and_save_model():
    # Load the trained model
    model = tf.keras.models.load_model('baseline_model.h5')

    # Load the eval dataset
    X_eval, y_eval = load_features_from_h5('processed_features.h5')

    # Reshape the eval features for CNN compatibility
    X_eval = X_eval.reshape((X_eval.shape[0], X_eval.shape[1], X_eval.shape[2], 1))

    # Evaluate the model
    loss, accuracy = model.evaluate(X_eval, y_eval)
    print(f'Evaluation Loss: {loss:.4f}, Evaluation Accuracy: {accuracy * 100:.2f}%')

    # Save model layers, weights, and evaluation results in a new HDF5 file
    with h5py.File('model_details.h5', 'w') as f:
        # Save model architecture
        model_json = model.to_json()
        f.create_dataset('model_architecture', data=model_json)

        # Save model weights
        for layer in model.layers:
            if layer.weights:
                layer_name = layer.name
                group = f.create_group(f"layer_weights/{layer_name}")
                for weight in layer.weights:
                    group.create_dataset(weight.name, data=weight.numpy())

        # Save evaluation metrics
        eval_group = f.create_group("evaluation_results")
        eval_group.create_dataset('loss', data=loss)
        eval_group.create_dataset('accuracy', data=accuracy)
    
    print("Model details and evaluation results saved to model_details.h5")


if __name__ == '__main__':
    evaluate_and_save_model()
