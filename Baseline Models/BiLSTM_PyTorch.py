import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, label_binarize
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from itertools import cycle
from torch.utils.data import TensorDataset, DataLoader

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. Load dataset
df = pd.read_csv("emotion_dataset_cleaned.csv")

# 2. Split features and labels
X = df.drop('label', axis=1).values.astype(np.float32)
y = df['label'].values

# 3. Encode labels
encoder = LabelEncoder()
y = encoder.fit_transform(y)

# 4. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.4, random_state=42, stratify=y
)

# 5. Reshape for LSTM input: (batch, seq_len, input_size)
X_train = np.expand_dims(X_train, axis=1)
X_test = np.expand_dims(X_test, axis=1)

# 6. Convert to tensors
X_train_tensor = torch.tensor(X_train)
X_test_tensor = torch.tensor(X_test)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

# 7. Dataset and Dataloader
batch_size = 32
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(test_dataset, batch_size=batch_size)

# 8. Define BiLSTM model
class BiLSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(BiLSTMClassifier, self).__init__()
        self.bilstm1 = nn.LSTM(input_size, hidden_size, batch_first=True, bidirectional=True)
        self.bn1 = nn.BatchNorm1d(2 * hidden_size)
        self.dropout1 = nn.Dropout(0.3)

        self.bilstm2 = nn.LSTM(2 * hidden_size, hidden_size, batch_first=True, bidirectional=True)
        self.bn2 = nn.BatchNorm1d(2 * hidden_size)
        self.dropout2 = nn.Dropout(0.3)

        self.fc1 = nn.Linear(2 * hidden_size, 64)
        self.dropout3 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        out, _ = self.bilstm1(x)
        out = self.dropout1(out)
        out = self.bn1(out[:, -1, :])

        out, _ = self.bilstm2(out.unsqueeze(1))
        out = self.dropout2(out[:, -1, :])
        out = self.bn2(out)

        out = F.relu(self.fc1(out))
        out = self.dropout3(out)
        return self.fc2(out)

# 9. Initialize model
input_size = X_train.shape[2]
num_classes = len(np.unique(y))
model = BiLSTMClassifier(input_size, 64, num_classes).to(device)

# 10. Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 11. Training loop
epochs = 100
train_loss_history, val_loss_history = [], []
train_acc_history, val_acc_history = [], []

for epoch in range(epochs):
    model.train()
    total_loss, correct = 0, 0

    for batch_X, batch_y in train_loader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)

        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == batch_y).sum().item()

    train_loss = total_loss / len(train_loader)
    train_acc = correct / len(train_loader.dataset)
    train_loss_history.append(train_loss)
    train_acc_history.append(train_acc)

    # Validation
    model.eval()
    val_loss, val_correct = 0, 0
    with torch.no_grad():
        for batch_X, batch_y in val_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            val_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            val_correct += (predicted == batch_y).sum().item()

    val_loss /= len(val_loader)
    val_acc = val_correct / len(val_loader.dataset)
    val_loss_history.append(val_loss)
    val_acc_history.append(val_acc)

    print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

# Save history
history_df = pd.DataFrame({
    'train_loss': train_loss_history,
    'val_loss': val_loss_history,
    'train_acc': train_acc_history,
    'val_acc': val_acc_history
})
history_df.to_csv("bilstm_training_history.csv", index=False)

# Accuracy Plot
plt.figure()
plt.plot(train_acc_history, label="Train Accuracy")
plt.plot(val_acc_history, label="Validation Accuracy")
plt.title("Accuracy Over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)
plt.savefig("bilstm_accuracy_history.png")
plt.show()

# Loss Plot
plt.figure()
plt.plot(train_loss_history, label="Train Loss")
plt.plot(val_loss_history, label="Validation Loss")
plt.title("Loss Over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.savefig("bilstm_loss_history.png")
plt.show()

# Evaluation
all_preds = []
all_probs = []
all_labels = []

with torch.no_grad():
    for batch_X, batch_y in val_loader:
        batch_X = batch_X.to(device)
        outputs = model(batch_X)
        probs = F.softmax(outputs, dim=1)
        preds = torch.argmax(probs, dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())
        all_labels.extend(batch_y.numpy())

# Classification Report
print("Classification Report BiLSTM:")
print(classification_report(all_labels, all_preds, target_names=encoder.classes_))

# Confusion Matrix
conf_matrix = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(10, 6))
sns.heatmap(conf_matrix / np.sum(conf_matrix, axis=0), annot=True, fmt=".2%", cmap="Blues",
            xticklabels=encoder.classes_, yticklabels=encoder.classes_)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.savefig("bilstm_confusion_matrix.png")
plt.show()

# ROC Curve
y_test_bin = label_binarize(all_labels, classes=range(num_classes))
y_score = np.array(all_probs)
fpr, tpr, roc_auc = {}, {}, {}

for i in range(num_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

fpr["micro"], tpr["micro"], _ = roc_curve(y_test_bin.ravel(), y_score.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

# Plot ROC
plt.figure()
colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'green', 'red', 'purple', 'brown'])
for i, color in zip(range(num_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color,
             label=f'ROC curve of class {encoder.classes_[i]} (area = {roc_auc[i]:0.2f})')

plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - BiLSTM')
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig("BiLSTM_ROC_Curve.png")
plt.show()
