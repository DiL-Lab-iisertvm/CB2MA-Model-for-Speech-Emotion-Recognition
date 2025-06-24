import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, label_binarize
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import seaborn as sns
from itertools import cycle

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load dataset
df = pd.read_csv("emotion_dataset_cleaned.csv")
X = df.drop('label', axis=1).values.astype(np.float32)
y = df['label'].values
encoder = LabelEncoder()
y = encoder.fit_transform(y)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.4, stratify=y, random_state=42
)
X_train = np.expand_dims(X_train, axis=-1)
X_test = np.expand_dims(X_test, axis=-1)

# Convert to tensors
X_train_tensor = torch.tensor(X_train)
X_test_tensor = torch.tensor(X_test)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

# DataLoader
batch_size = 32
train_loader = DataLoader(TensorDataset(X_train_tensor, y_train_tensor), batch_size=batch_size, shuffle=True)
val_loader = DataLoader(TensorDataset(X_test_tensor, y_test_tensor), batch_size=batch_size)

# Attention Layer
class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.attn = nn.Linear(hidden_size * 2, 1)

    def forward(self, encoder_outputs):
        attn_weights = torch.softmax(self.attn(encoder_outputs), dim=1)
        context = torch.sum(attn_weights * encoder_outputs, dim=1)
        return context

# Model
class CNNBiLSTMAttention(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(CNNBiLSTMAttention, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(32)
        self.pool1 = nn.MaxPool1d(2)

        self.conv2 = nn.Conv1d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm1d(64)
        self.pool2 = nn.MaxPool1d(2)

        self.bilstm = nn.LSTM(64, 64, batch_first=True, bidirectional=True)
        self.attn = Attention(64)
        self.fc1 = nn.Linear(128, 64)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = x.permute(0, 2, 1)  # (batch, 1, seq_len) -> (batch, seq_len, 1)
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = x.permute(0, 2, 1)  # (batch, seq_len, channels)
        output, _ = self.bilstm(x)
        attn_output = self.attn(output)
        x = self.dropout(F.relu(self.fc1(attn_output)))
        x = self.fc2(x)
        return x

# Model setup
input_dim = X_train.shape[1]
num_classes = len(np.unique(y))
model = CNNBiLSTMAttention(input_dim, num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training
epochs = 50
train_acc_hist, val_acc_hist = [], []
train_loss_hist, val_loss_hist = [], []

for epoch in range(epochs):
    model.train()
    running_loss, correct = 0, 0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        preds = model(xb)
        loss = criterion(preds, yb)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        correct += (torch.argmax(preds, dim=1) == yb).sum().item()
    train_loss = running_loss / len(train_loader)
    train_acc = correct / len(train_loader.dataset)
    train_loss_hist.append(train_loss)
    train_acc_hist.append(train_acc)

    # Validation
    model.eval()
    val_loss, val_correct = 0, 0
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(device), yb.to(device)
            preds = model(xb)
            val_loss += criterion(preds, yb).item()
            val_correct += (torch.argmax(preds, dim=1) == yb).sum().item()
    val_loss /= len(val_loader)
    val_acc = val_correct / len(val_loader.dataset)
    val_loss_hist.append(val_loss)
    val_acc_hist.append(val_acc)
    print(f"Epoch {epoch+1}: Train Acc={train_acc:.4f}, Val Acc={val_acc:.4f}")

# Save history
pd.DataFrame({
    'train_loss': train_loss_hist,
    'val_loss': val_loss_hist,
    'train_acc': train_acc_hist,
    'val_acc': val_acc_hist
}).to_csv("CNN_BiLSTM_Attention_training_history.csv", index=False)

# Accuracy and Loss Plots
plt.plot(train_acc_hist, label="Train Acc")
plt.plot(val_acc_hist, label="Val Acc")
plt.title("Accuracy")
plt.legend()
plt.savefig("CNN_BiLSTM_Attention_accuracy.png")
plt.clf()

plt.plot(train_loss_hist, label="Train Loss")
plt.plot(val_loss_hist, label="Val Loss")
plt.title("Loss")
plt.legend()
plt.savefig("CNN_BiLSTM_Attention_loss.png")
plt.clf()

# Evaluation
model.eval()
all_preds, all_probs, all_true = [], [], []

with torch.no_grad():
    for xb, yb in val_loader:
        xb = xb.to(device)
        out = model(xb)
        prob = F.softmax(out, dim=1)
        all_probs.extend(prob.cpu().numpy())
        all_preds.extend(torch.argmax(prob, dim=1).cpu().numpy())
        all_true.extend(yb.numpy())

print("Classification Report:")
print(classification_report(all_true, all_preds, target_names=encoder.classes_))

conf_matrix = confusion_matrix(all_true, all_preds)
sns.heatmap(conf_matrix / np.sum(conf_matrix, axis=0), annot=True, fmt='.2%', cmap="Blues",
            xticklabels=encoder.classes_, yticklabels=encoder.classes_)
plt.title("Confusion Matrix")
plt.savefig("CNN_BiLSTM_Attention_confusion_matrix.png")
plt.clf()

# ROC Curve
y_test_bin = label_binarize(all_true, classes=range(num_classes))
y_score = np.array(all_probs)
fpr, tpr, roc_auc = {}, {}, {}

for i in range(num_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

fpr["micro"], tpr["micro"], _ = roc_curve(y_test_bin.ravel(), y_score.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'green', 'red', 'purple', 'brown'])
for i, color in zip(range(num_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color,
             label=f'Class {encoder.classes_[i]} (area = {roc_auc[i]:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.title("ROC Curve")
plt.legend()
plt.savefig("CNN_BiLSTM_Attention_ROC.png")
