import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------
# 1. Load Dataset
# -----------------------------

vocab = {
    "LOGIN": 0,
    "LOGOUT": 1,
    "FILE_ACCESS": 2,
    "PRIV_ESC": 3,
    "DATA_EXFIL": 4,
    "FAILED_LOGIN": 5,
    "PASSWORD_RESET": 6,
    "NONE": 7
}

num_classes = 4
vocab_size = len(vocab)

def load_dataset(file):
    df = pd.read_csv(file, header=None)
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values

    X_encoded = []
    for row in X:
        X_encoded.append([vocab[token] for token in row])

    return torch.tensor(X_encoded).long(), torch.tensor(y).long()

X, y = load_dataset("cyber_multiclass_dataset.csv")

split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# -----------------------------
# 2. Positional Encoding
# -----------------------------

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=50):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

# -----------------------------
# 3. Transformer Model
# -----------------------------

class TransformerClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        d_model = 64

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=4,
            dim_feedforward=128,
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=2
        )

        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        x = self.pos_encoder(x)
        x = self.transformer(x)
        x = x.mean(dim=1)
        return self.fc(x)

# -----------------------------
# 4. Training
# -----------------------------

model = TransformerClassifier()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 15
loss_values = []

for epoch in range(epochs):

    optimizer.zero_grad()
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()

    loss_values.append(loss.item())
    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

# -----------------------------
# 5. Evaluation
# -----------------------------

with torch.no_grad():
    outputs = model(X_test)
    predictions = torch.argmax(outputs, dim=1)
    accuracy = (predictions == y_test).float().mean()

print("Test Accuracy:", accuracy.item())

# -----------------------------
# 6. Confusion Matrix
# -----------------------------

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test.numpy(), predictions.numpy())

plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.savefig("confusion_matrix.png")
plt.show()
