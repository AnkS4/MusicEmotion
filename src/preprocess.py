from librosa import load
from librosa.onset import onset_strength
from librosa.feature import rhythm, spectral_centroid, spectral_rolloff, zero_crossing_rate
import soundfile as sf
import numpy as np


class AudioFeatureExtractor:
    """
    A class for extracting audio features from a given audio file.
    """

    def __init__(self, file_path: str):
        """
        Initialize the AudioFeatureExtractor with the audio file path.

        Args:
            file_path (str): Path to the audio file.
        """
        self.file_path = file_path
        self.audio_data, self.sample_rate = self._load_audio()

    def _load_audio(self):
        """
        Load the audio file and fetch its sample rate.

        Returns:
            tuple: Audio time series data and sample rate.
        """
        try:
            info = sf.info(self.file_path)
            audio, sr = load(self.file_path, sr=info.samplerate, mono=True)  # Use sample rate from sf.info
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
        return rhythm.tempo(onset_envelope=onset_env, sr=self.sample_rate)[0]

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


if __name__ == "__main__":
    sample_file = "../data/33796__yewbic__ambience03.wav"

    feature_extractor = AudioFeatureExtractor(file_path=sample_file)
    features = feature_extractor.extract_features()

    print("Extracted Features:", features)

