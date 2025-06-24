import os
import numpy as np
import librosa
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import seaborn as sns
import random

# =========================== Feature Extraction ===========================
def extract_features(file_path, n_mfcc=40):
    try:
        audio, sample_rate = librosa.load(file_path, res_type='kaiser_fast')
        mfccs = np.mean(librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=n_mfcc).T, axis=0)
        mel = np.mean(librosa.feature.melspectrogram(y=audio, sr=sample_rate).T, axis=0)
        zcr = np.mean(librosa.feature.zero_crossing_rate(audio).T, axis=0)
        chroma = np.mean(librosa.feature.chroma_stft(y=audio, sr=sample_rate).T, axis=0)
        contrast = np.mean(librosa.feature.spectral_contrast(y=audio, sr=sample_rate).T, axis=0)
        rms = np.mean(librosa.feature.rms(y=audio).T, axis=0)
        pitches, magnitudes = librosa.core.piptrack(y=audio, sr=sample_rate)
        pitch_scaled = np.mean(pitches.T, axis=0)
        combined = np.hstack([mfccs, mel, zcr, chroma, contrast, rms, pitch_scaled])
        return combined
    except Exception as e:
        print(f"Error extracting features from {file_path}: {e}")
        return None

def extract_features_from_audio(audio, sample_rate, n_mfcc=40):
    try:
        mfccs = np.mean(librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=n_mfcc).T, axis=0)
        mel = np.mean(librosa.feature.melspectrogram(y=audio, sr=sample_rate).T, axis=0)
        zcr = np.mean(librosa.feature.zero_crossing_rate(audio).T, axis=0)
        chroma = np.mean(librosa.feature.chroma_stft(y=audio, sr=sample_rate).T, axis=0)
        contrast = np.mean(librosa.feature.spectral_contrast(y=audio, sr=sample_rate).T, axis=0)
        rms = np.mean(librosa.feature.rms(y=audio).T, axis=0)
        pitches, magnitudes = librosa.core.piptrack(y=audio, sr=sample_rate)
        pitch_scaled = np.mean(pitches.T, axis=0)
        combined = np.hstack([mfccs, mel, zcr, chroma, contrast, rms, pitch_scaled])
        return combined
    except Exception as e:
        print(f"Error in augmented feature extraction: {e}")
        return None

def augment_data(file_path, n_augmentations=5):
    try:
        audio, sample_rate = librosa.load(file_path, res_type='kaiser_fast')
        augmented = []
        for _ in range(n_augmentations):
            pitch_shift = random.uniform(-2, 2)
            aug_audio = librosa.effects.pitch_shift(audio, sr=sample_rate, n_steps=pitch_shift)
            time_stretch = random.uniform(0.8, 1.2)
            aug_audio = librosa.effects.time_stretch(aug_audio, rate=time_stretch)
            augmented.append((aug_audio, sample_rate))
        return augmented
    except Exception as e:
        print(f"Error augmenting {file_path}: {e}")
        return []

def load_dataset(dataset_path, label, features, labels, augment=True):
    if not os.path.exists(dataset_path):
        print(f"Path not found: {dataset_path}")
        return
    for file in os.listdir(dataset_path):
        if file.endswith(".wav"):
            file_path = os.path.join(dataset_path, file)
            print(f"Processing: {file_path}")
            feature = extract_features(file_path)
            if feature is not None:
                features.append(feature)
                labels.append(label)
                if augment:
                    augmented = augment_data(file_path)
                    for aug_audio, sr in augmented:
                        aug_feature = extract_features_from_audio(aug_audio, sr)
                        if aug_feature is not None:
                            features.append(aug_feature)
                            labels.append(label)

# =========================== Paths ===========================
tess_path = r"D:\IISER TVM\Projects\RamaKrishnan Kerala University\Journal Paper Writing\Raw Dataset\TESS Toronto emotional speech set data"
ravdess_path = r"D:\IISER TVM\Projects\RamaKrishnan Kerala University\Journal Paper Writing\Raw Dataset\ravdess\audio_speech_actors_01-24"
savee_path = r"D:\IISER TVM\Projects\RamaKrishnan Kerala University\Journal Paper Writing\Raw Dataset\savee_dataset\ALL"  # fixed typo

print("TESS path exists:", os.path.exists(tess_path))
print("RAVDESS path exists:", os.path.exists(ravdess_path))
print("SAVEe path exists:", os.path.exists(savee_path))


# =========================== Data Loading ===========================
augment = True
features = []
labels = []

# --- TESS ---
emotions_tess = {
    'OAF_angry': 'angry', 'OAF_disgust': 'disgust', 'OAF_fear': 'fear',
    'OAF_happy': 'happy', 'OAF_neutral': 'neutral', 'OAF_sad': 'sad',
    'OAF_Pleasant_surprise': 'surprised', 'YAF_angry': 'angry',
    'YAF_disgust': 'disgust', 'YAF_fear': 'fear', 'YAF_happy': 'happy',
    'YAF_neutral': 'neutral', 'YAF_sad': 'sad', 'YAF_pleasant_surprised': 'surprised'
}

for folder, label in emotions_tess.items():
    dir_path = os.path.join(tess_path, folder)
    load_dataset(dir_path, label, features, labels, augment)

#--- RAVDESS ---
emotion_map_ravdess = {
    '01': 'neutral', '02': 'calm', '03': 'happy', '04': 'sad',
    '05': 'angry', '06': 'fearful', '07': 'disgust', '08': 'surprised'
}

for root, dirs, files in os.walk(ravdess_path):
    for file in files:
        if file.endswith(".wav"):
            emotion_id = file.split("-")[2]
            emotion = emotion_map_ravdess.get(emotion_id)
            if emotion:
                file_path = os.path.join(root, file)
                print(f"Processing: {file_path}")
                feature = extract_features(file_path)
                if feature is not None:
                    features.append(feature)
                    labels.append(emotion)
                    if augment:
                        for aug_audio, sr in augment_data(file_path):
                            aug_feature = extract_features_from_audio(aug_audio, sr)
                            if aug_feature is not None:
                                features.append(aug_feature)
                                labels.append(emotion)

#--- SAVEe ---
emotion_map_savee = {
    'a': 'angry',
    'd': 'disgust',
    'f': 'fear',
    'h': 'happy',
    'n': 'neutral',
    'sa': 'surprised',
    's': 'sad'
}

for file in os.listdir(savee_path):
    if file.endswith(".wav"):
        file_path = os.path.join(savee_path, file)
        basename = os.path.splitext(file)[0].lower()

        # Check for "sa" (surprise) first, then others
        if 'sa' in basename:
            emotion_key = 'sa'
        else:
            emotion_key = basename[3]  # DC_a01 → 4th character = a

        if emotion_key in emotion_map_savee:
            emotion = emotion_map_savee[emotion_key]
            print(f"Processing SAVEe: {file_path} → {emotion}")
            feature = extract_features(file_path)
            if feature is not None:
                features.append(feature)
                labels.append(emotion)
                if augment:
                    for aug_audio, sr in augment_data(file_path):
                        aug_feature = extract_features_from_audio(aug_audio, sr)
                        if aug_feature is not None:
                            features.append(aug_feature)
                            labels.append(emotion)
        else:
            print(f"Unrecognized emotion code '{emotion_key}' in file: {file}")

# =========================== Final Preprocessing ===========================
print("Total features:", len(features))
print("Total labels:", len(labels))

if len(features) == 0:
    print("No features extracted. Please check your dataset paths and audio files.")
    exit()

# ---------------- Save to CSV ----------------
print("Saving features and labels to CSV...")

features_df = pd.DataFrame(features)
features_df['label'] = labels
features_df.to_csv("combined_emotion_dataset.csv", index=False)

print("Done! CSV saved as combined_emotion_dataset.csv")
#######################################################################

