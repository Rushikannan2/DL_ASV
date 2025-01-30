import librosa
import os
import numpy as np
import logging
import pandas as pd
from tqdm import tqdm  # Progress bar for longer processing
import h5py  # For storing features in HDF5 format

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

def load_audio(file_path, sr=16000):
    """
    Load an audio file and return the waveform and sample rate.
    """
    try:
        audio, sample_rate = librosa.load(file_path, sr=sr)
        logger.info(f"Loaded audio from {file_path} with sample rate {sample_rate}.")
        return audio, sample_rate
    except Exception as e:
        logger.error(f"Error loading {file_path}: {e}")
        return None, None

def extract_mfcc_features(audio, sample_rate, n_mfcc=13, max_time_steps=159):
    """
    Extract MFCC features from audio and pad/truncate to a fixed number of time steps.
    """
    if audio is None:
        return None
    try:
        mfcc = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=n_mfcc)
        logger.info(f"Extracted MFCC features with shape: {mfcc.shape}")

        # Pad or truncate to a fixed number of time steps
        if mfcc.shape[1] < max_time_steps:
            mfcc = np.pad(mfcc, ((0, 0), (0, max_time_steps - mfcc.shape[1])), mode='constant')
        elif mfcc.shape[1] > max_time_steps:
            mfcc = mfcc[:, :max_time_steps]

        return mfcc.T  # Transpose to have time steps as rows
    except Exception as e:
        logger.error(f"Error extracting MFCC from audio: {e}")
        return None

def preprocess_audio_files(dataset_path, max_time_steps=159):
    """
    Preprocess all audio files in a given directory and extract features.
    """
    features = []
    labels = []

    if not os.path.exists(dataset_path):
        logger.error(f"Dataset path does not exist: {dataset_path}")
        return features, labels

    for root, _, files in os.walk(dataset_path):
        for file_name in tqdm(files, desc="Processing files"):
            if file_name.endswith(('.wav', '.flac')):
                file_path = os.path.join(root, file_name)

                # Load audio
                audio, sample_rate = load_audio(file_path)
                if audio is None:
                    continue

                # Extract MFCC features
                mfcc_features = extract_mfcc_features(audio, sample_rate, max_time_steps=max_time_steps)
                if mfcc_features is None:
                    continue

                # Label based on filename convention
                label = 0 if 'bonafide' in file_name.lower() else 1

                features.append(mfcc_features)
                labels.append(label)

    logger.info(f"Extracted {len(features)} feature sets with corresponding labels.")
    return features, labels

def save_features_to_h5(features, labels, output_file='processed_features.h5'):
    """
    Save features and labels to an HDF5 file.
    """
    try:
        with h5py.File(output_file, "w") as hf:
            hf.create_dataset("features", data=np.array(features), compression="gzip")
            hf.create_dataset("labels", data=np.array(labels), compression="gzip")
        logger.info(f"Features and labels successfully saved to '{output_file}'.")
    except Exception as e:
        logger.error(f"Error saving features to HDF5: {e}")

def load_features_from_h5(file_path):
    """
    Load features and labels from an HDF5 file.
    """
    try:
        with h5py.File(file_path, "r") as hf:
            features = np.array(hf["features"])
            labels = np.array(hf["labels"])
        logger.info(f"Loaded features and labels from '{file_path}'.")
        return features, labels
    except Exception as e:
        logger.error(f"Error loading HDF5 file: {e}")
        return None, None


# Example usage
dataset_path = r"C:\Users\Rushi\OneDrive\Desktop\ASVspoof_LA_Project\datasets\train"
features, labels = preprocess_audio_files(dataset_path)

# Save extracted features and labels to an HDF5 file
save_features_to_h5(features, labels, "processed_features.h5")

# Load the saved features for later training
X_data, y_data = load_features_from_h5("processed_features.h5")

