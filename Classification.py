import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from reservoirpy.nodes import Reservoir, Ridge
from reservoirpy import set_seed, verbosity
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import OneHotEncoder

# Reproducibility
set_seed(42)
verbosity(0)

# --- Functions ---
def generate_pattern(label, length):
    t = np.linspace(0, 2 * np.pi, length)
    if label == 0:
        return np.sin(t)
    elif label == 1:
        return np.sign(np.sin(t))
    elif label == 2:
        return 2 * (t / (2 * np.pi)) - 1
    else:
        raise ValueError("Unknown Label.")

# --- 1. Generate data ---
n_patterns = 100
pattern_length = 20
n_classes = 3

X_list = []
y_list = []

for _ in range(n_patterns):
    label = np.random.randint(0, n_classes)
    pattern = generate_pattern(label, pattern_length).reshape(-1, 1)
    X_list.append(pattern)
    y_list.extend([label] * pattern_length)

X = np.vstack(X_list)
y = np.array(y_list)

# --- 2. Training-/Testdata ---
split_idx = int(0.7 * len(X))
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

# --- 3. Define model ---
reservoir = Reservoir(100, lr=0.7, sr=0.95)
readout = Ridge(ridge=1e-6)
esn_model = reservoir >> readout

# --- 4. One-Hot-Encoding for classification ---
encoder = OneHotEncoder(sparse_output=False)
y_train_oh = encoder.fit_transform(y_train.reshape(-1, 1))

# --- 5. Training ---
esn_model = esn_model.fit(X_train, y_train_oh, warmup=10, reset=True)

# --- 6. Prediction ---
y_pred_oh = esn_model.run(X_test)
y_pred = np.argmax(y_pred_oh, axis=1)

# --- 7. Combined Plot ---
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Left: Time Series
axes[0].set_title("Input-Signal & Klassenvorhersage")
axes[0].plot(X_test[:200], label="Input-Signal")
axes[0].plot(y_test[:200], label="Ground Truth", linestyle="--")
axes[0].plot(y_pred[:200], label="Prediction", linestyle=":")
axes[0].set_xlabel("Time")
axes[0].set_ylabel("Wert / Klasse")
axes[0].legend()

# Right: Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
            xticklabels=["Sinus", "Rechteck", "Dreieck"],
            yticklabels=["Sinus", "Rechteck", "Dreieck"],
            ax=axes[1])
axes[1].set_title("Konfusionsmatrix")
axes[1].set_xlabel("Vorhergesagte Klasse")
axes[1].set_ylabel("Tatsächliche Klasse")

plt.tight_layout()
plt.show()

# --- 8. Klassifikationsbericht (optional) ---
print("Classification Report:\n")
print(classification_report(y_test, y_pred, target_names=["Sinus", "Rechteck", "Dreieck"]))