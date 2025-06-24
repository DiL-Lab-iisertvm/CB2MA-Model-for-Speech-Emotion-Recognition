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
import seaborn as sns
import matplotlib.pyplot as plt

##############################################################################
# Load the cleaned dataset
df = pd.read_csv('emotion_dataset_cleaned.csv')  # Make sure it's in your working directory
#############################################################################
# Plot label distribution
plt.figure(figsize=(10, 6))
sns.countplot(data=df, x='label', order=df['label'].value_counts().index, palette='viridis')
# Add labels and title
plt.title('Emotion Label Distribution')
plt.xlabel('Emotion Class')
plt.ylabel('Number of Samples')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("data_distribution.png")
plt.show()
print(df['label'].value_counts())
##############t-SNE#######################################
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# Separate features and labels
X = df.drop('label', axis=1)
y = df['label']

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply t-SNE
tsne = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=42)
X_tsne = tsne.fit_transform(X_scaled)

# Create a DataFrame for plotting
tsne_df = pd.DataFrame({
    'TSNE1': X_tsne[:, 0],
    'TSNE2': X_tsne[:, 1],
    'Label': y
})

# Visualize using Seaborn
plt.figure(figsize=(10, 7))
sns.scatterplot(data=tsne_df, x='TSNE1', y='TSNE2', hue='Label', palette='tab10', s=70)
plt.title('t-SNE Visualization of Combined SER Emotion Dataset')
plt.legend(loc='best', bbox_to_anchor=(1.05, 1), borderaxespad=0.)
plt.tight_layout()
plt.savefig("tsne.png")
plt.show()

############################################################################################

mean_features_by_label = df.groupby('label').mean()
plt.figure(figsize=(14, 6))
sns.heatmap(mean_features_by_label.T, cmap='YlGnBu', annot=False)
plt.title("Mean Feature Values by Emotion")
plt.tight_layout()
plt.savefig("Mean Feature.png")
plt.show()

#########################################################################################