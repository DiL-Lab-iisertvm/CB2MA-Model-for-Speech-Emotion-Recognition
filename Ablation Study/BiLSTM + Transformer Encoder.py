import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, label_binarize
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
import seaborn as sns
from itertools import cycle

# 1. Load dataset
df = pd.read_csv("emotion_dataset_cleaned.csv")
X = df.drop("label", axis=1).values
y = df["label"].values

# 2. Encode labels
encoder = LabelEncoder()
y = encoder.fit_transform(y)
num_classes = len(np.unique(y))

# 3. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.4, stratify=y, random_state=42
)

# 4. Reshape for sequence models
X_train = X_train.reshape(X_train.shape[0], -1, 1)
X_test = X_test.reshape(X_test.shape[0], -1, 1)

# 5. Convert to torch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32)

# 6. Define the BiLSTM + Transformer model
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * -(np.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class BiLSTMTransformer(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes, num_heads=4, ff_dim=128):
        super().__init__()
        self.bilstm = nn.LSTM(input_dim, hidden_dim, num_layers=1, batch_first=True, bidirectional=True)
        self.pos_encoder = PositionalEncoding(d_model=hidden_dim * 2)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim * 2,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=0.1,
            activation="relu",
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        out, _ = self.bilstm(x)
        out = self.pos_encoder(out)
        out = self.transformer_encoder(out)
        out = out.mean(dim=1)
        return self.classifier(out)

# 7. Training setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BiLSTMTransformer(input_dim=1, hidden_dim=128, num_classes=num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 8. Training loop
train_acc, val_acc = [], []
for epoch in range(2):
    model.train()
    correct, total, loss_total = 0, 0, 0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        output = model(X_batch)
        loss = criterion(output, y_batch)
        loss.backward()
        optimizer.step()
        _, predicted = torch.max(output.data, 1)
        total += y_batch.size(0)
        correct += (predicted == y_batch).sum().item()
        loss_total += loss.item()
    acc = correct / total
    train_acc.append(acc)
    print(f"Epoch {epoch+1}, Train Accuracy: {acc:.4f}, Loss: {loss_total:.4f}")

# 9. Evaluation
model.eval()
y_true, y_pred, y_score = [], [], []
with torch.no_grad():
    for X_batch, y_batch in test_loader:
        X_batch = X_batch.to(device)
        output = model(X_batch)
        probs = torch.softmax(output, dim=1).cpu().numpy()
        preds = np.argmax(probs, axis=1)
        y_pred.extend(preds)
        y_true.extend(y_batch.numpy())
        y_score.extend(probs)

# 10. Confusion Matrix
conf_matrix = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(10, 6))
sns.heatmap(conf_matrix / np.sum(conf_matrix, axis=1, keepdims=True), annot=True, cmap="Blues", fmt=".2f",
            xticklabels=encoder.classes_, yticklabels=encoder.classes_)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig("BiLSTM_Transformer_confusion_matrix.png")
plt.show()

# 11. Classification Report
print("Classification Report:")
print(classification_report(y_true, y_pred, target_names=encoder.classes_))

# 12. ROC Curve
y_test_bin = label_binarize(y_true, classes=list(range(num_classes)))
y_score = np.array(y_score)
fpr, tpr, roc_auc = {}, {}, {}
for i in range(num_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

plt.figure(figsize=(10, 6))
colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'green', 'red', 'purple', 'brown'])
for i, color in zip(range(num_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=2, label=f"Class {encoder.classes_[i]} (AUC = {roc_auc[i]:.2f})")

plt.plot([0, 1], [0, 1], "k--")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("BiLSTM + Transformer Encoder ROC Curve")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig("BiLSTM_Transformer_ROC_Curve.png")
plt.show()
