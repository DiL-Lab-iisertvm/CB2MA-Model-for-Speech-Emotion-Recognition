import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# Step 1: Load the combined dataset
df = pd.read_csv('combined_emotion_dataset.csv')

# Step 2: Combine 'fearful' and 'fear' into a single class
df['label'] = df['label'].replace({'fearful': 'fear'})

# Step 3: Exclude 'calm' class
df_filtered = df[df['label'] != 'calm']

# Step 4: Save the cleaned dataset to a new CSV file
df_filtered.to_csv('emotion_dataset_cleaned.csv', index=False)

print("Cleaned dataset saved as 'emotion_dataset_cleaned.csv'")


