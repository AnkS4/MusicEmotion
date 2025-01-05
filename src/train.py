from preprocess import AudioFeatureExtractor, flatten_features
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import os
import glob


random_state = 123

# Function to process multiple audio files and extract features
def process_audio_files(file_paths, labels):
    """
    Process multiple audio files, extract features, and return a DataFrame with labels.

    Args:
        file_paths (list): List of audio file paths.
        labels (list): Corresponding labels for the audio files.

    Returns:
        pd.DataFrame: DataFrame with flattened features and labels.
    """
    all_features = []
    for file_path, label in zip(file_paths, labels):
        try:
            feature_extractor = AudioFeatureExtractor(file_path=file_path)
            features = feature_extractor.extract_features()
            flat_features = flatten_features(features)
            flat_features["label"] = label
            all_features.append(flat_features)
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
    return pd.DataFrame(all_features)


audio_folder = "../data/"
audio_files = glob.glob(os.path.join(audio_folder, "*.wav"))
labels = []  # Example labels (e.g., 0 for 'class A', 1 for 'class B')

# Extract features and create a DataFrame
data = process_audio_files(audio_files, labels)

# Prepare features and labels for training
X = data.drop(columns=["label"])  # Features
y = data["label"]  # Labels

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)

# Train a simple model (Random Forest Classifier in this example)
model = RandomForestClassifier(random_state=random_state)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:")
print(classification_report(y_test, y_pred))
