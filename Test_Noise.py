import numpy as np
import matplotlib.pyplot as plt
from reservoirpy.nodes import Reservoir, Ridge
from reservoirpy import set_seed, verbosity

# Reproducibility
import time
set_seed(int(time.time()))
verbosity(0)

# 1. Define model
reservoir = Reservoir(100, lr=0.5, sr=0.9)
readout = Ridge(ridge=1e-7)
esn_model = reservoir >> readout

# 2. Generate sine data
X_clean = np.sin(np.linspace(0, 6 * np.pi, 100)).reshape(-1, 1)

# 3. Add noise (e.g., Gaussian noise with std=0.1)
noise = np.random.normal(loc=0.0, scale=0.1, size=X_clean.shape)
X_noisy = X_clean + noise

# 4. Training data (only noisy part)
X_train = X_noisy[:50]
Y_train = X_noisy[1:51]

# 5. Train model
esn_model = esn_model.fit(X_train, Y_train, warmup=10, reset=True)

# 6. Autoregressive prediction
n_steps = 49  # Number of steps to predict
initial_input = X_noisy[50].reshape(1, -1)
predictions = []

current_input = initial_input
for _ in range(n_steps):
    pred = esn_model.run(current_input)
    predictions.append(pred)
    current_input = pred

Y_pred = np.vstack(predictions)
Y_true = X_clean[51:51+n_steps]

# 7. Plotting
plt.figure(figsize=(10, 4))
plt.title("Noisy sine – autoregressive prediction")
plt.xlabel("$t$")
plt.ylabel(r"$\sin(t)$")

# Original (clean)
plt.plot(np.arange(len(X_clean)), X_clean, label="Original (clean)", color="gray", alpha=0.4)

# Training data (only front part)
plt.plot(np.arange(50), X_noisy[:50], label="Training data (noisy)", color="orange", alpha=0.6)

# Prediction
plt.plot(np.arange(51, 51 + n_steps), Y_pred, label="Prediction", color="blue")

# Ground truth for comparison
plt.plot(np.arange(51, 51 + n_steps), Y_true, label="Target (clean)", color="red", linestyle="dashed")

plt.axvline(x=50, linestyle="--", color="black", alpha=0.5, label="Start of prediction")
plt.legend()
plt.tight_layout()
plt.show()

from sklearn.metrics import mean_squared_error

# --- Metrics ---
mse = mean_squared_error(Y_true, Y_pred)
rmse = np.sqrt(mse)

print("\n🔎 Prediction results:\n")
print(f"✅ MSE:  {mse:.6f}")
print(f"✅ RMSE: {rmse:.6f}\n")

# --- Comparison of some values ---
print("📊 Comparison: Prediction vs. Ground Truth")
print("Index |   Prediction   |     Ground Truth   |    Delta")
print("---------------------------------------------------------")
for i in range(min(10, len(Y_true))):
    pred_val = Y_pred[i, 0]
    true_val = Y_true[i, 0]
    delta = pred_val - true_val
    print(f"{i:5d} | {pred_val:13.6f} | {true_val:17.6f} | {delta:10.6f}")