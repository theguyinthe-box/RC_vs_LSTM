import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from reservoirpy.nodes import Reservoir, Ridge
from reservoirpy import set_seed
from itertools import product
from mpl_toolkits.mplot3d import Axes3D
import time

# Rössler-System definieren
def rossler(t, u, a=0.2, b=0.2, c=5.7):
    x, y, z = u
    return [-y - z, x + a * y, b + z * (x - c)]

# Rössler-Daten generieren
def generate_rossler_data():
    t_span = (0, 200)
    t_eval = np.arange(0, 200, 0.02)
    u0 = [0.1, 0.0, 0.0]
    sol = solve_ivp(rossler, t_span, u0, t_eval=t_eval)
    return sol.y.T  # shape: (time, 3)

# Trainings- und Vorhersagefunktion mit Zeitmessung
def train_and_generate(params, X_train, Y_train, scaler, predict_len, test, test_scaled):
    set_seed(42)
    
    reservoir = Reservoir(units=params['units'], sr=params['sr'], lr=params['lr'])
    readout = Ridge(ridge=params['ridge'])
    esn = reservoir >> readout

    # Training
    start_fit = time.perf_counter()
    esn = esn.fit(X_train, Y_train, warmup=100, reset=True)
    end_fit = time.perf_counter()
    print(f"      ⏱ Training duration: {end_fit - start_fit:.4f} seconds")

    # Vorhersage
    outputs = []
    last_input = X_train[-1].reshape(1, -1)
    step_times = []

    start_pred = time.perf_counter()
    for step in range(predict_len):
        t0 = time.perf_counter()
        pred = esn.run(last_input)
        t1 = time.perf_counter()
        
        step_times.append(t1 - t0)
        outputs.append(pred.ravel())
        last_input = pred
    end_pred = time.perf_counter()

    step_times_np = np.array(step_times)
    print(f"\nAverage prediction time per step: {step_times_np.mean() * 1e3:.6f} ms")
    print(f"Minimum prediction time per step: {step_times_np.min() * 1e3:.6f} ms at step {step_times_np.argmin()}")
    print(f"Maximum prediction time per step: {step_times_np.max() * 1e3:.6f} ms at step {step_times_np.argmax()}")
    print(f"      ⏱ Prediction duration: {end_pred - start_pred:.4f} seconds")

    output_scaled = np.array(outputs)
    output = scaler.inverse_transform(output_scaled)

    mse = mean_squared_error(test, output)
    mse_scaled = mean_squared_error(test_scaled, output_scaled)

    return mse, mse_scaled, output, output_scaled

# Daten vorbereiten
data = generate_rossler_data()
shift = 300
train_len = 5000
predict_len = 500

X = data[shift:shift+train_len]
Y = data[shift+1:shift+train_len+1]
test = data[shift+train_len:shift+train_len+predict_len]

# Skalieren
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
Y_scaled = scaler.transform(Y)
test_scaled = scaler.transform(test)

# Parameter-Raum
param_grid = {
    'sr': [1.5],
    'lr': [0.2],
    'units': [2200],
    'ridge': [1e-2]
}

# Beste Parameter suchen
best_mse = float('inf')
best_mse_scaled = float('inf')
best_params = None
best_output = None
best_output_scaled = None
best_idx = None

total_start = time.perf_counter()

for idx, (sr, lr, units, ridge) in enumerate(product(param_grid['sr'], param_grid['lr'], param_grid['units'], param_grid['ridge'])):
    params = {'sr': sr, 'lr': lr, 'units': units, 'ridge': ridge}
    print(f"\n[{idx}] Testing params: {params}")
    
    start_model = time.perf_counter()
    mse, mse_scaled, output, output_scaled = train_and_generate(params, X_scaled, Y_scaled, scaler, predict_len, test, test_scaled)
    end_model = time.perf_counter()
    
    print(f"   --> MSE: {mse:.6f} | MSE_scaled: {mse_scaled:.6f}")
    print(f"   ⏱ Total time for this model: {end_model - start_model:.2f} seconds")

    if mse < best_mse:
        print("   ✅ New best model found!")
        best_mse = mse
        best_mse_scaled = mse_scaled
        best_params = params
        best_output = output
        best_output_scaled = output_scaled
        best_idx = idx

total_end = time.perf_counter()
print(f"\n⏱ Total runtime for all models: {total_end - total_start:.2f} seconds")

# MSE pro Dimension berechnen
mse_x = mean_squared_error(test[:, 0], best_output[:, 0])
mse_y = mean_squared_error(test[:, 1], best_output[:, 1])
mse_z = mean_squared_error(test[:, 2], best_output[:, 2])

mse_scaled_x = mean_squared_error(test_scaled[:, 0], best_output_scaled[:, 0])
mse_scaled_y = mean_squared_error(test_scaled[:, 1], best_output_scaled[:, 1])
mse_scaled_z = mean_squared_error(test_scaled[:, 2], best_output_scaled[:, 2])

# Visualisierung
fig = plt.figure(figsize=(12, 8))

ax1 = fig.add_subplot(221)
ax1.plot(best_output[:, 0], label="pred x")
ax1.plot(test[:, 0], '--', label="true x")
ax1.set_title(f"x(t)\nMSE unscaled: {mse_x:.6f} | scaled: {mse_scaled_x:.6f}")
ax1.legend()

ax2 = fig.add_subplot(222)
ax2.plot(best_output[:, 1], label="pred y")
ax2.plot(test[:, 1], '--', label="true y")
ax2.set_title(f"y(t)\nMSE unscaled: {mse_y:.6f} | scaled: {mse_scaled_y:.6f}")
ax2.legend()

ax3 = fig.add_subplot(223)
ax3.plot(best_output[:, 2], label="pred z")
ax3.plot(test[:, 2], '--', label="true z")
ax3.set_title(f"z(t)\nMSE unscaled: {mse_z:.6f} | scaled: {mse_scaled_z:.6f}")
ax3.legend()

ax4 = fig.add_subplot(224, projection='3d')
ax4.plot(*best_output.T, label="predicted")
ax4.plot(*test.T, linestyle="--", label="true")
ax4.set_title(
    f"Rössler attractor\nMSE unscaled (total): {best_mse:.6f}\nMSE scaled (total): {best_mse_scaled:.6f}"
)
ax4.legend()

plt.suptitle(f"Best Hyperparameters: sr={best_params['sr']}, lr={best_params['lr']}, "
             f"units={best_params['units']}, ridge={best_params['ridge']}\n"
             f"Model Number: {best_idx}")
plt.tight_layout()
plt.show()