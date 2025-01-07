from preprocess import AudioFeatureExtractor, flatten_features
import pandas as pd


feature_data = []
annotations = pd.read_csv("data/static_annotations_averaged.csv", dtype={'song_id': str})  # Load the annotations

for _, row in annotations.iterrows():
    song_id = row["song_id"]
    file_path = f"/data/audios/{song_id}.mp3"  # Path to the audio file

    print(f"Extracting features for {song_id}.mp3 ...")
    # Extract features for the song
    feature_extractor = AudioFeatureExtractor(file_path)
    features = feature_extractor.extract_features()
    flat_features = flatten_features(features)

    # Add the extracted features and labels to the dataset
    flat_features["valence_mean"] = row["valence_mean"]
    flat_features["arousal_mean"] = row["arousal_mean"]
    feature_data.append(flat_features)

# Create a DataFrame for training
dataset = pd.DataFrame(feature_data)
dataset.to_csv("data/features.csv")
