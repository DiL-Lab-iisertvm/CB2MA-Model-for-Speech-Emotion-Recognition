import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import cycle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, label_binarize
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

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

# 4. Reshape for CNN input
X_train = X_train.reshape(X_train.shape[0], -1, 1)
X_test = X_test.reshape(X_test.shape[0], -1, 1)

# 5. Convert to tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32)

# 6. Attention layer
class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super(Attention, self).__init__()
        self.attn = nn.Linear(hidden_dim * 2, 1)

    def forward(self, rnn_output):
        # rnn_output: (batch, seq_len, hidden_dim*2)
        attn_weights = self.attn(rnn_output).squeeze(-1)  # (batch, seq_len)
        attn_weights = torch.softmax(attn_weights, dim=1)
        context = torch.sum(rnn_output * attn_weights.unsqueeze(-1), dim=1)  # (batch, hidden_dim*2)
        return context, attn_weights

# 7. CNN + BiGRU + Attention model
class CNN_BiGRU_Attention(nn.Module):
    def __init__(self, input_dim, cnn_out_channels, gru_hidden, num_classes):
        super(CNN_BiGRU_Attention, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=input_dim, out_channels=cnn_out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )
        self.bigru = nn.GRU(
            input_size=cnn_out_channels, hidden_size=gru_hidden,
            batch_first=True, bidirectional=True
        )
        self.attention = Attention(hidden_dim=gru_hidden)
        self.classifier = nn.Sequential(
            nn.Linear(gru_hidden * 2, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        # x: (batch, seq_len, 1)
        x = x.permute(0, 2, 1)              # (batch, 1, seq_len)
        x = self.cnn(x)                     # (batch, cnn_channels, seq_len//2)
        x = x.permute(0, 2, 1)              # (batch, seq_len//2, cnn_channels)
        gru_out, _ = self.bigru(x)          # (batch, seq_len//2, hidden*2)
        context, _ = self.attention(gru_out)
        out = self.classifier(context)
        return out

# 8. Training setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNN_BiGRU_Attention(input_dim=1, cnn_out_channels=64, gru_hidden=128, num_classes=num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 9. Training loop
for epoch in range(10):
    model.train()
    total, correct, total_loss = 0, 0, 0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        out = model(xb)
        loss = criterion(out, yb)
        loss.backward()
        optimizer.step()
        preds = out.argmax(dim=1)
        correct += (preds == yb).sum().item()
        total += yb.size(0)
        total_loss += loss.item()
    print(f"Epoch {epoch+1}, Accuracy: {correct/total:.4f}, Loss: {total_loss:.4f}")

# 10. Evaluation
model.eval()
y_true, y_pred, y_score = [], [], []
with torch.no_grad():
    for xb, yb in test_loader:
        xb = xb.to(device)
        logits = model(xb)
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        preds = np.argmax(probs, axis=1)
        y_pred.extend(preds)
        y_true.extend(yb.numpy())
        y_score.extend(probs)

# 11. Confusion Matrix
conf_matrix = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(10, 6))
sns.heatmap(conf_matrix / np.sum(conf_matrix, axis=1, keepdims=True),
            annot=True, fmt=".2f", cmap="Blues",
            xticklabels=encoder.classes_, yticklabels=encoder.classes_)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix: CNN + BiGRU + Attention")
plt.tight_layout()
plt.savefig("CNN_BiGRU_Attention_confusion_matrix.png")
plt.show()

# 12. Classification Report
print("Classification Report:")
print(classification_report(y_true, y_pred, target_names=encoder.classes_))

# 13. ROC Curve
y_test_bin = label_binarize(y_true, classes=range(num_classes))
y_score = np.array(y_score)
fpr, tpr, roc_auc = {}, {}, {}
for i in range(num_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

plt.figure(figsize=(10, 6))
colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'green', 'red', 'purple', 'brown'])
for i, color in zip(range(num_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=2,
             label=f"Class {encoder.classes_[i]} (AUC = {roc_auc[i]:.2f})")
plt.plot([0, 1], [0, 1], "k--", lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - CNN + BiGRU + Attention")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig("CNN_BiGRU_Attention_ROC_Curve.png")
plt.show()
