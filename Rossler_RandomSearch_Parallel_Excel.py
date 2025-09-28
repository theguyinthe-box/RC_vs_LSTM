import jax
import jax.numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from reservoirpy.jax.nodes import Reservoir, Ridge
from reservoirpy import set_seed
from itertools import product
from joblib import Parallel, delayed
from mpl_toolkits.mplot3d import Axes3D
import time
import random
import pandas as pd

# Define the Rössler system
def rossler(t, u, a=0.2, b=0.2, c=5.7):
    x, y, z = u
    return [-y - z, x + a * y, b + z * (x - c)]

# Generate Rössler data
def generate_rossler_data():
    t_span = (0, 200)
    t_eval = np.arange(0, 200, 0.02)
    u0 = [0.1, 0.0, 0.0]
    sol = solve_ivp(rossler, t_span, u0, t_eval=t_eval)
    return sol.y.T  # shape: (time, 3)

# Prepare data
data = generate_rossler_data()
shift = 300
train_len = 5000
predict_len = 500

X = data[shift:shift+train_len]
Y = data[shift+1:shift+train_len+1]
test = data[shift+train_len:shift+train_len+predict_len]

# Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
Y_scaled = scaler.transform(Y)
test_scaled = scaler.transform(test)

# Training and prediction function
def train_and_generate(params, X_train, Y_train, scaler, predict_len, test, test_scaled, idx):
    set_seed(42)

    reservoir = Reservoir(
        units=params['units'],
        sr=params['sr'],
        lr=params['lr'],
        input_scaling=params['input_scaling']
    )
    readout = Ridge(ridge=params['ridge'])
    esn = reservoir >> readout

    start_fit = time.perf_counter()
    esn = esn.fit(X_train, Y_train, warmup=100)
    end_fit = time.perf_counter()
    fit_time = end_fit - start_fit

    outputs = []
    last_input = X_train[-1].reshape(1, -1)
    step_times = []

    start_pred = time.perf_counter()
    for _ in range(predict_len):
        t0 = time.perf_counter()
        pred = esn.run(last_input)
        t1 = time.perf_counter()
        step_times.append(t1 - t0)
        outputs.append(pred.ravel())
        last_input = pred
    end_pred = time.perf_counter()

    output_scaled = np.array(outputs)
    output = scaler.inverse_transform(output_scaled)

    mse = mean_squared_error(test, output)
    mse_scaled = mean_squared_error(test_scaled, output_scaled)

    return {
        "mse": mse,
        "mse_scaled": mse_scaled,
        "output": output,
        "output_scaled": output_scaled,
        "params": params,
        "idx": idx,
        "fit_time": fit_time,
        "pred_time": end_pred - start_pred,
        "step_times": np.array(step_times)
    }

# Parameter space
param_grid = {
    'sr': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0],
    'lr': [0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    'units': [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000, 2100, 2200, 2300, 2400, 2500, 2600, 2700, 2800, 2900, 3000, 3100, 3200, 3300, 3400, 3500, 3600, 3700, 3800, 3900, 4000, 4100, 4200, 4300, 4400, 4500, 4600, 4700, 4800, 4900, 5000, 5100, 5200, 5300, 5400, 5500, 5600, 5700, 5800, 5900, 6000, 6100, 6200, 6300, 6400, 6500, 6600, 6700, 6800, 6900, 7000, 7100, 7200, 7300, 7400, 7500, 7600, 7700, 7800, 7900, 8000, 8100, 8200, 8300, 8400, 8500, 8600, 8700, 8800, 8900, 9000, 9100, 9200, 9300, 9400, 9500, 9600, 9700, 9800, 9900, 10000],
    'ridge': [1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1],
    'input_scaling': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
}

# Number of random combinations
n_samples = 200  # Adjust depending on runtime/performance

# Generate random parameter combinations
param_combinations = random.sample(
    list(product(
        param_grid['sr'],
        param_grid['lr'],
        param_grid['units'],
        param_grid['ridge'],
        param_grid['input_scaling']
    )),
    n_samples
)


# Parallel execution
total_start = time.perf_counter()
results = Parallel(n_jobs=-1)(
    delayed(train_and_generate)(
        {
            'sr': sr,
            'lr': lr,
            'units': units,
            'ridge': ridge,
            'input_scaling': input_scaling
        },
        X_scaled, Y_scaled, scaler, predict_len, test, test_scaled, idx
    )
    for idx, (sr, lr, units, ridge, input_scaling) in enumerate(param_combinations)
)

total_end = time.perf_counter()

# Find best model
best_result = min(results, key=lambda x: x["mse"])

# Excel file
excel_file = "rc_rossler_results.xlsx"

# Convert results to DataFrame
df = pd.DataFrame([
    {
        "idx": r["idx"],
        "units": r["params"]["units"],
        "sr": r["params"]["sr"],
        "lr": r["params"]["lr"],
        "ridge": r["params"]["ridge"],
        "input_scaling": r["params"]["input_scaling"],
        "mse": r["mse"],
        "mse_scaled": r["mse_scaled"],
        "fit_time_s": r["fit_time"],
        "pred_time_s": r["pred_time"],
        "avg_step_time_ms": r["step_times"].mean() * 1e3,
        "min_step_time_ms": r["step_times"].min() * 1e3,
        "max_step_time_ms": r["step_times"].max() * 1e3
    }
    for r in results
])

if "idx" in df.columns:
    df = df.drop(columns=["idx"])

# Try loading existing results
try:
    existing_df = pd.read_excel(excel_file)
    combined_df = pd.concat([existing_df, df], ignore_index=True)
    combined_df = combined_df.sort_values(
    by=["mse"]
).drop_duplicates(subset=["units", "sr", "lr", "ridge", "input_scaling"], keep="first")
    df = combined_df  # work with the combined DataFrame from now on
except FileNotFoundError:
    pass

# Sort
df = df.sort_values(
    by=["mse", "avg_step_time_ms", "units", "sr", "lr", "ridge", "input_scaling"]
).reset_index(drop=True)

# Export in Excel
df.to_excel(excel_file, index=False, engine="openpyxl")

print(f"Results successfully saved in: {excel_file}")

# Results overview
for r in results:
    print(f"[{r['idx']}] MSE: {r['mse']:.6f} | MSE_scaled: {r['mse_scaled']:.6f} | Params: {r['params']}")
    print(f"     ⏱ Training time: {r['fit_time']:.4f}s | Prediction time: {r['pred_time']:.4f}s")
    print(f"     → Avg pred step: {r['step_times'].mean()*1e3:.4f} ms | min: {r['step_times'].min()*1e3:.4f} ms | max: {r['step_times'].max()*1e3:.4f} ms")

print(f"\n✅ Best model is #{best_result['idx']} with MSE: {best_result['mse']:.6f} and scaled MSE: {best_result['mse_scaled']:.6f}")
print(f"⏱ Total runtime for all models: {total_end - total_start:.2f} seconds")

# Prediction time details for best model
avg_step_time_ms = best_result["step_times"].mean() * 1e3
min_step_time_ms = best_result["step_times"].min() * 1e3
max_step_time_ms = best_result["step_times"].max() * 1e3

print(f"\n🔍 Prediction time details for best model #{best_result['idx']}:")
print(f"     → Total prediction time: {best_result['pred_time']:.4f} seconds")
print(f"     → Avg step time: {avg_step_time_ms:.4f} ms")
print(f"     → Min step time: {min_step_time_ms:.4f} ms")
print(f"     → Max step time: {max_step_time_ms:.4f} ms")

# MSE per dimension
mse_x = mean_squared_error(test[:, 0], best_result["output"][:, 0])
mse_y = mean_squared_error(test[:, 1], best_result["output"][:, 1])
mse_z = mean_squared_error(test[:, 2], best_result["output"][:, 2])

mse_scaled_x = mean_squared_error(test_scaled[:, 0], best_result["output_scaled"][:, 0])
mse_scaled_y = mean_squared_error(test_scaled[:, 1], best_result["output_scaled"][:, 1])
mse_scaled_z = mean_squared_error(test_scaled[:, 2], best_result["output_scaled"][:, 2])

# Visualization
fig = plt.figure(figsize=(12, 8))

ax1 = fig.add_subplot(221)
ax1.plot(best_result["output"][:, 0], label="pred x")
ax1.plot(test[:, 0], '--', label="true x")
ax1.set_title(f"x(t)\nMSE unscaled: {mse_x:.6f} | scaled: {mse_scaled_x:.6f}")
ax1.legend()

ax2 = fig.add_subplot(222)
ax2.plot(best_result["output"][:, 1], label="pred y")
ax2.plot(test[:, 1], '--', label="true y")
ax2.set_title(f"y(t)\nMSE unscaled: {mse_y:.6f} | scaled: {mse_scaled_y:.6f}")
ax2.legend()

ax3 = fig.add_subplot(223)
ax3.plot(best_result["output"][:, 2], label="pred z")
ax3.plot(test[:, 2], '--', label="true z")
ax3.set_title(f"z(t)\nMSE unscaled: {mse_z:.6f} | scaled: {mse_scaled_z:.6f}")
ax3.legend()

ax4 = fig.add_subplot(224, projection='3d')
ax4.plot(*best_result["output"].T, label="predicted")
ax4.plot(*test.T, linestyle="--", label="true")
ax4.set_title(
    f"Rössler attractor\nMSE unscaled (total): {best_result['mse']:.6f}\nMSE scaled (total): {best_result['mse_scaled']:.6f}"
)
ax4.legend()

plt.suptitle(
    f"Best Hyperparameters: {best_result['params']} | Model #{best_result['idx']}\n"
    f"Avg Prediction Step Time: {avg_step_time_ms:.4f} ms"
)
plt.tight_layout()
plt.show()

# Additional visualization for prediction time vs. reservoir size
mse_list = [r["mse"] for r in results]
units_list = [r["params"]["units"] for r in results]
avg_pred_time_ms = [r["step_times"].mean() * 1e3 for r in results]  # in ms

plt.figure(figsize=(10, 6))
sc = plt.scatter(units_list, avg_pred_time_ms, c=mse_list, cmap='RdYlGn_r', alpha=0.7, edgecolors='k')
plt.xlabel("Number of Reservoir Units")
plt.ylabel("Average Prediction Time per Step (ms)")
plt.title("Reservoir Size vs. Prediction Time (colored by MSE)")
plt.colorbar(sc, label="MSE")
plt.grid(True)
plt.tight_layout()
plt.show()