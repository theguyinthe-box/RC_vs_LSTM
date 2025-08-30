import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_squared_error
from scipy.integrate import solve_ivp
from sklearn.preprocessing import StandardScaler
from itertools import product
import numpy as np
import matplotlib.pyplot as plt
import random
import time
import pandas as pd
import os
from tqdm import tqdm



# --- Rössler System Definition ---
def rossler(t, u, a=0.2, b=0.2, c=5.7):
    x, y, z = u
    return [-y - z, x + a * y, b + z * (x - c)]

# --- Generate Data ---
def generate_rossler_data():
    t_span = (0, 200)
    t_eval = np.arange(0, 200, 0.02)
    u0 = [0.1, 0.0, 0.0]
    sol = solve_ivp(rossler, t_span, u0, t_eval=t_eval)
    return sol.y.T  # shape: (time, 3)

# --- Prepare Data ---
data = generate_rossler_data()
shift = 300
train_len = 5000
predict_len = 500

X = data[shift:shift + train_len]
Y = data[shift + 1:shift + train_len + 1]
test = data[shift + train_len:shift + train_len + predict_len]

# --- Scaling ---
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
Y_scaled = scaler.transform(Y)
test_scaled = scaler.transform(test)

# Hyperparameter grid
param_grid = {
    'hidden_size': [32, 64, 128, 256, 384, 512, 768],
    'num_layers': [1, 2, 3, 4],
    'lr': [1e-2, 5e-3, 1e-3, 5e-4, 1e-4, 5e-5, 1e-5],
    'sequence_length': [5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
}

n_samples = 5  # Adjust based on runtime

param_combinations = random.sample(
    list(product(param_grid['hidden_size'], param_grid['num_layers'], param_grid['lr'], param_grid['sequence_length'])),
    n_samples
)

INPUT_SIZE = 3
OUTPUT_SIZE = 3
EPOCHS = 50
BATCH_SIZE = 64

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Sequence creation function
def create_sequences(X, Y, seq_len):
    x_seq, y_seq = [], []
    for i in range(len(X) - seq_len):
        x_seq.append(X[i:i+seq_len])
        y_seq.append(Y[i+seq_len-1])
    return np.array(x_seq), np.array(y_seq)

# LSTM Model
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

# Training + Prediction
def train_and_predict_lstm(params, X_scaled, Y_scaled, test, test_scaled, scaler, idx):
    # Fix seed
    torch.manual_seed(42)
    np.random.seed(42)
    
    sequence_length = params['sequence_length']
    x_seq, y_seq = create_sequences(X_scaled, Y_scaled, sequence_length)
    dataset = TensorDataset(torch.Tensor(x_seq), torch.Tensor(y_seq))
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    tqdm.write("Instantiating model and moving to device...")
    model = LSTMModel(INPUT_SIZE, params['hidden_size'], OUTPUT_SIZE, params['num_layers']).to(device)
    tqdm.write("Model is now on device:", device)

    optimizer = torch.optim.Adam(model.parameters(), lr=params['lr'])
    criterion = nn.MSELoss()

    # Training
    model.train()
    start_fit = time.perf_counter()
    for epoch in range(EPOCHS):
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
    end_fit = time.perf_counter()

    # Prediction
    model.eval()
    predictions = []
    step_times = []
    last_seq = torch.Tensor(X_scaled[-sequence_length:]).unsqueeze(0).to(device)

    start_pred = time.perf_counter()
    with torch.no_grad():
        for _ in range(predict_len):
            t0 = time.perf_counter()
            pred = model(last_seq)
            t1 = time.perf_counter()
            step_times.append(t1 - t0)

            predictions.append(pred.cpu().numpy().flatten())
            pred_input = pred.unsqueeze(1)
            last_seq = torch.cat((last_seq[:, 1:], pred_input), dim=1)
    end_pred = time.perf_counter()

    pred_scaled = np.array(predictions)
    pred_unscaled = scaler.inverse_transform(pred_scaled)

    mse = mean_squared_error(test, pred_unscaled)
    mse_scaled = mean_squared_error(test_scaled, pred_scaled)

    return {
        "mse": mse,
        "mse_scaled": mse_scaled,
        "params": params,
        "output": pred_unscaled,
        "output_scaled": pred_scaled,
        "fit_time": end_fit - start_fit,
        "pred_time": end_pred - start_pred,
        "step_times": np.array(step_times),
        "idx": idx
    }

# Run all combinations
results = []
total_start = time.perf_counter()

print(f"Starting LSTM Random Search with {n_samples} combinations (parallel)...\n")
total_start = time.perf_counter()

results = []
print(f"Starting LSTM Random Search with {n_samples} combinations (sequential + progress bar)...\n")
total_start = time.perf_counter()

for idx, (hidden, layers, lr, seq_len) in tqdm(enumerate(param_combinations), total=n_samples, desc="Random Search Progress"):
    params = {
        'hidden_size': hidden,
        'num_layers': layers,
        'lr': lr,
        'sequence_length': seq_len
    }
    tqdm.write(f"➡️  Running combination #{idx+1}/{n_samples}: {params}")
    
    result = train_and_predict_lstm(params, X_scaled, Y_scaled, test, test_scaled, scaler, idx)
    results.append(result)

total_end = time.perf_counter()


# Sort & show best
best_result = min(results, key=lambda r: r['mse'])

# Excel-Dateiname
excel_file = "lstm_rossler_results.xlsx"

# Ergebnisse in DataFrame umwandeln
df = pd.DataFrame([
    {
    "idx": r["idx"],
    "mse": r["mse"],
    "mse_scaled": r["mse_scaled"],
    "avg_pred_time_per_step_ms": r["step_times"].mean() * 1e3,
    "hidden_size": r["params"]["hidden_size"],
    "num_layers": r["params"]["num_layers"],
    "lr": r["params"]["lr"],
    "sequence_length": r["params"]["sequence_length"]
    }
    for r in results
])

if "idx" in df.columns:
    df = df.drop(columns=["idx"])

# Versuche Datei zu laden, oder beginne neu
try:
    existing_df = pd.read_excel(excel_file)
    combined_df = pd.concat([existing_df, df], ignore_index=True)
    combined_df = combined_df.sort_values(
    by=["mse"]
).drop_duplicates(subset=["hidden_size", "num_layers", "lr", "sequence_length"], keep="first")
    df = combined_df  # arbeite ab jetzt mit dem kombinierten
except FileNotFoundError:
    pass

# Sortierung nach Wunsch anwenden
df = df.sort_values(
    by=["mse", "avg_pred_time_per_step_ms", "hidden_size", "num_layers", "lr", "sequence_length"]
).reset_index(drop=True)

# Exportiere in Excel
df.to_excel(excel_file, index=False, engine="openpyxl")

print(f"📊 Ergebnisse erfolgreich gespeichert in: {excel_file}")

for r in results:
    print(f"[{r['idx']}] MSE: {r['mse']:.6f} | MSE_scaled: {r['mse_scaled']:.6f} | Params: {r['params']}")
    print(f"     ⏱ Training time: {r['fit_time']:.4f}s | Prediction time: {r['pred_time']:.4f}s")
    print(f"     → Avg pred step: {r['step_times'].mean()*1e3:.4f} ms | min: {r['step_times'].min()*1e3:.4f} ms | max: {r['step_times'].max()*1e3:.4f} ms")

print(f"\n✅ Best model is #{best_result['idx']} with MSE: {best_result['mse']:.6f} and scaled MSE: {best_result['mse_scaled']:.6f}")
print(f"⏱ Total runtime for all models: {total_end - total_start:.2f} seconds")

# MSE pro Dimension berechnen
mse_x = mean_squared_error(test[:, 0], best_result["output"][:, 0])
mse_y = mean_squared_error(test[:, 1], best_result["output"][:, 1])
mse_z = mean_squared_error(test[:, 2], best_result["output"][:, 2])

mse_scaled_x = mean_squared_error(test_scaled[:, 0], best_result["output_scaled"][:, 0])
mse_scaled_y = mean_squared_error(test_scaled[:, 1], best_result["output_scaled"][:, 1])
mse_scaled_z = mean_squared_error(test_scaled[:, 2], best_result["output_scaled"][:, 2])

# Ergebnisse im gleichen Stil wie bei RC ausgeben
for r in results:
    print(f"[{r['idx']}] MSE: {r['mse']:.6f} | MSE_scaled: {r['mse_scaled']:.6f} | Params: {r['params']}")
    print(f"     ⏱ Training time: {r['fit_time']:.4f}s | Prediction time: {r['pred_time']:.4f}s")
    print(f"     → Avg pred step: {r['step_times'].mean()*1e3:.4f} ms | min: {r['step_times'].min()*1e3:.4f} ms | max: {r['step_times'].max()*1e3:.4f} ms")

print(f"\n✅ Best model is #{best_result['idx']} with MSE: {best_result['mse']:.6f} and scaled MSE: {best_result['mse_scaled']:.6f}")
print(f"⏱ Total runtime for all models: {total_end - total_start:.2f} seconds")

# --- Plot ---
fig = plt.figure(figsize=(12, 8))

ax1 = fig.add_subplot(221)
ax1.plot(best_result["output"][:, 0], label="LSTM pred x")
ax1.plot(test[:, 0], '--', label="true x")
ax1.set_title(f"x(t)\nMSE unscaled: {mse_x:.6f} | scaled: {mse_scaled_x:.6f}")
ax1.legend()

ax2 = fig.add_subplot(222)
ax2.plot(best_result["output"][:, 1], label="LSTM pred y")
ax2.plot(test[:, 1], '--', label="true y")
ax2.set_title(f"y(t)\nMSE unscaled: {mse_y:.6f} | scaled: {mse_scaled_y:.6f}")
ax2.legend()

ax3 = fig.add_subplot(223)
ax3.plot(best_result["output"][:, 2], label="LSTM pred z")
ax3.plot(test[:, 2], '--', label="true z")
ax3.set_title(f"z(t)\nMSE unscaled: {mse_z:.6f} | scaled: {mse_scaled_z:.6f}")
ax3.legend()

ax4 = fig.add_subplot(224, projection='3d')
ax4.plot(*best_result["output"].T, label="predicted")
ax4.plot(*test.T, linestyle="--", label="true")
ax4.set_title(
    f"LSTM Rössler attractor\nMSE unscaled (total): {best_result['mse']:.6f}\nMSE scaled (total): {best_result['mse_scaled']:.6f}"
)
ax4.legend()

plt.suptitle(f"Best Hyperparameters: {best_result['params']} | Model #{best_result['idx']}")
plt.tight_layout()
plt.show()

# --- Scatter-Plot: Prediction Time vs Hidden Size, MSE als Farbe ---
hidden_sizes = [r['params']['hidden_size'] for r in results]
avg_pred_times = [r['step_times'].mean() * 1e3 for r in results]  # in ms
mse_vals = [r['mse'] for r in results]

plt.figure(figsize=(10, 6))
sc = plt.scatter(hidden_sizes, avg_pred_times, c=mse_vals, cmap='RdYlGn_r', edgecolors='k', alpha=0.7)
plt.xlabel("Hidden Size")
plt.ylabel("Average Prediction Time per Step (ms)")
plt.title("LSTM Hidden Size vs. Prediction Time (colored by MSE)")
plt.colorbar(sc, label="MSE")
plt.grid(True)
plt.tight_layout()
plt.show()