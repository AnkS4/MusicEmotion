from librosa import load
from librosa.onset import onset_strength
from librosa.feature import spectral_centroid, spectral_rolloff, zero_crossing_rate, mfcc, chroma_stft, spectral_contrast
from librosa.feature import tempo
import soundfile as sf
import numpy as np
import pandas as pd
import os


class AudioFeatureExtractor:
    """
    A class for extracting audio features from a given audio file.
    """

    def __init__(self, file_path: str = None):
        """
        Initialize the AudioFeatureExtractor with the audio file path.

        Args:
            file_path (str): Path to the audio file.
        """
        if file_path is None:
            # Dynamically construct the absolute path to the default file
            current_dir = os.path.dirname(os.path.abspath(__file__))  # Get the directory of this script
            self.file_path = os.path.join(current_dir, "../data/33796__yewbic__ambience03.wav")
        else:
            self.file_path = file_path
        self.audio_data, self.sample_rate = self._load_audio()

    def _load_audio(self):
        """
        Load the audio file and fetch its sample rate.

        Returns:
            tuple: Audio time series data and sample rate.
        """
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"Audio file not found: {self.file_path}")

        try:
            info = sf.info(self.file_path)
            mono = info.channels == 1  # Set mono=True if the audio has only one channel
            audio, sr = load(self.file_path, sr=info.samplerate, mono=mono)
            return audio, sr
        except Exception as e:
            raise RuntimeError(f"Error loading audio file {self.file_path}: {e}")

    def extract_features(self):
        """
        Extract audio features and handle errors gracefully.

        Returns:
            dict: Extracted audio features.
        """
        feature_extractors = {
            "tempo": self._extract_tempo,
            "energy": self._extract_energy,
            "spectral_centroid": self._extract_spectral_centroid,
            "spectral_rolloff": self._extract_spectral_rolloff,
            "zero_crossing_rate": self._extract_zero_crossing_rate,
            "mfcc": self._extract_mfcc,
            "chroma": self._extract_chroma,
            "spectral_contrast": self._extract_spectral_contrast
        }

        extracted_features = {}
        for feature_name, extractor in feature_extractors.items():
            try:
                extracted_features[feature_name] = extractor()
            except Exception as e:
                extracted_features[feature_name] = None
                print(f"Error calculating {feature_name}: {e}")

        return extracted_features

    def _extract_tempo(self):
        """
        Calculate the tempo of the audio.

        Returns:
            float: Tempo in beats per minute.
        """
        onset_env = onset_strength(y=self.audio_data, sr=self.sample_rate)
        return tempo(onset_envelope=onset_env, sr=self.sample_rate)[0]

    def _extract_energy(self):
        """
        Calculate the energy of the audio.

        Returns:
            float: Energy of the audio signal.
        """
        return np.sum(np.square(self.audio_data)) / len(self.audio_data)

    def _extract_spectral_centroid(self):
        """
        Calculate the spectral centroid of the audio.

        Returns:
            float: Spectral centroid.
        """
        return np.mean(spectral_centroid(y=self.audio_data, sr=self.sample_rate))

    def _extract_spectral_rolloff(self):
        """
        Calculate the spectral rolloff of the audio.

        Returns:
            float: Spectral rolloff.
        """
        return np.mean(spectral_rolloff(y=self.audio_data, sr=self.sample_rate))

    def _extract_zero_crossing_rate(self):
        """
        Calculate the zero crossing rate of the audio.

        Returns:
            float: Zero crossing rate.
        """
        return np.mean(zero_crossing_rate(self.audio_data))
        
    def _extract_mfcc(self):
        """
        Extract MFCC features.
        
        Returns:
        np.ndarray: Mean MFCC values.
        """
        return np.mean(mfcc(y=self.audio_data, sr=self.sample_rate, n_mfcc=13), axis=1)
        
    def _extract_chroma(self):
        """
        Extract chroma features.
        
        Returns:
        np.ndarray: Mean chroma features.
        """
        return np.mean(chroma_stft(y=self.audio_data, sr=self.sample_rate), axis=1)
        
    def _extract_spectral_contrast(self):
        """
        Extract spectral contrast.
        
        Returns:
        np.ndarray: Mean spectral contrast.
        """
        return np.mean(spectral_contrast(y=self.audio_data, sr=self.sample_rate), axis=1)

def flatten_features(features: dict) -> dict:
    flat_features = {}
    for key, value in features.items():
        if isinstance(value, np.ndarray):  # Handle arrays (e.g., mfcc, chroma, spectral_contrast, etc.)
            if value.ndim == 1:  # If it's a 1D array, keep it as-is
                flat_features[key] = np.mean(value)
            elif value.ndim == 2:  # If it's a 2D array (time-series), summarize it
                for i, row in enumerate(value):
                    flat_features[f"{key}_mean_{i}"] = np.mean(row)
                    flat_features[f"{key}_std_{i}"] = np.std(row)
            else:
                raise ValueError(f"Unexpected shape for feature {key}: {value.shape}")
        else:  # Handle scalar values
            flat_features[key] = value
    return flat_features


if __name__ == "__main__":
    feature_extractor = AudioFeatureExtractor()
    features = feature_extractor.extract_features()
    flat_features = flatten_features(features)
    features_df = pd.DataFrame([flat_features])

    print(features_df)
