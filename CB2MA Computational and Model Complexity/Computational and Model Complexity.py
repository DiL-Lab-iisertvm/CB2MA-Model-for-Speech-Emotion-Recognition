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
# 1. Load dataset
df = pd.read_csv("emotion_dataset_cleaned.csv")


# 2. Split features and labels
X = df.drop('label', axis=1).values
y = df['label'].values

# 3. Encode labels
encoder = LabelEncoder()
y = encoder.fit_transform(y)

# 4. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.4, random_state=42, stratify=y
)

# 5. Reshape for CNN input
X_train = X_train[..., np.newaxis]
X_test = X_test[..., np.newaxis]

# 6. One-hot encode labels
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# 7. Print data shapes
print(f"X_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"y_test shape: {y_test.shape}")

# 8. Build CNN model
input_shape = X_train.shape[1:]  # (feature_dim, 1)
num_classes = y_train.shape[1]

#############################################################################################

import tensorflow as tf
from tensorflow.keras.layers import Input, Conv1D, BatchNormalization, MaxPooling1D, Bidirectional, LSTM, GRU, Dense, \
    Dropout, Add, LayerNormalization, MultiHeadAttention, Layer, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau


# Custom Attention Layer
class Attention(tf.keras.layers.Layer):
    def _init_(self, **kwargs):
        super(Attention, self)._init_(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(name='att_weight', shape=(input_shape[-1], input_shape[-1]),
                                 initializer='glorot_uniform', trainable=True)
        self.b = self.add_weight(name='att_bias', shape=(input_shape[-1],), initializer='zeros', trainable=True)
        self.u = self.add_weight(name='att_u', shape=(input_shape[-1],), initializer='glorot_uniform', trainable=True)
        super(Attention, self).build(input_shape)

    def call(self, x):
        # Score computation
        u_t = tf.tanh(tf.tensordot(x, self.W, axes=1) + self.b)
        score = tf.tensordot(u_t, self.u, axes=1)

        # Softmax over scores
        att_weights = tf.nn.softmax(score, axis=1)

        # Weighted sum of hidden states
        weighted_input = x * tf.expand_dims(att_weights, -1)
        return tf.reduce_sum(weighted_input, axis=1)

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[-1]


# Transformer Encoder function
def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
    x = MultiHeadAttention(key_dim=head_size, num_heads=num_heads)(inputs, inputs)
    x = Dropout(dropout)(x)
    x = Add()([x, inputs])
    x = LayerNormalization(epsilon=1e-6)(x)

    x_ff = Dense(ff_dim, activation="relu")(x)
    x_ff = Dropout(dropout)(x_ff)
    x_ff = Dense(inputs.shape[-1])(x_ff)
    x_ff = Add()([x_ff, x])
    x_ff = LayerNormalization(epsilon=1e-6)(x_ff)
    return x_ff


# Define the model
inputs = Input(shape=(X_train.shape[1], 1))

# Add Conv1D layers
x = Conv1D(64, kernel_size=5, strides=1, padding='same', activation='relu')(inputs)
x = BatchNormalization()(x)
x = MaxPooling1D(pool_size=2)(x)

x = Conv1D(128, kernel_size=5, strides=1, padding='same', activation='relu')(x)
x = BatchNormalization()(x)
x = MaxPooling1D(pool_size=2)(x)

x = Conv1D(256, kernel_size=5, strides=1, padding='same', activation='relu')(x)
x = BatchNormalization()(x)
x = MaxPooling1D(pool_size=2)(x)

# Add BiLSTM layer
x = Bidirectional(LSTM(128, return_sequences=True))(x)
x = Dropout(0.3)(x)

# Add Transformer Encoder layer
x = transformer_encoder(x, head_size=128, num_heads=8, ff_dim=512, dropout=0.3)

# Add BiGRU layer
x = Bidirectional(GRU(128, return_sequences=True))(x)
x = Dropout(0.3)(x)

# Add Attention layer
x = Attention()(x)

# Add Dense layers
x = Dense(128, activation='relu')(x)
x = Dropout(0.3)(x)

outputs = Dense(y_train.shape[1], activation='softmax')(x)

# Compile the model
model = Model(inputs, outputs)

# --- Calculate FLOPs ---
layer_names = []
layer_flops = []

for layer in model.layers:
    try:
        flops = tf.profiler.experimental.get_stats_for_node_def(
            tf.compat.v1.Session().graph.as_graph_def(),
            node_name=layer.name,
            step=None,
            profile_type='flops'
        )
        layer_flops.append(flops.total_float_ops)
    except:
        # Simple approximation
        flops = np.sum([np.prod(w.shape) for w in layer.weights]) * 2
        layer_flops.append(flops)
    layer_names.append(layer.name)

# --- Plot FLOPs per layer ---
plt.figure(figsize=(18, 12))
plt.bar(layer_names, layer_flops, color='skyblue')
plt.xlabel("Layers")
plt.ylabel("Computational Complexity")
plt.title("Computational Complexity of Each Layer in the Model")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("CB2MA_Computational_Complexity.png")
plt.show()

###########################################################################################

# Model Complexity Plot
def plot_model_complexity(model):
    layers = [layer for layer in model.layers]
    num_params = [layer.count_params() for layer in layers]
    layer_names = [layer.name for layer in layers]

    plt.figure(figsize=(12, 6))
    plt.barh(layer_names, num_params)
    plt.xlabel('Number of Parameters')
    plt.ylabel('Layers')
    plt.title('Model Complexity')
    plt.tight_layout()
    plt.savefig("CB2MA_Model_Complexity.png")
    plt.show()

plot_model_complexity(model)