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

# Define the Lorenz system
def lorenz(t, u, sigma=10., rho=28., beta=8/3):
    x, y, z = u
    return [
        sigma * (y - x),
        x * (rho - z) - y,
        x * y - beta * z
    ]

# Generate Lorenz data
def generate_lorenz_data():
    t_span = (0, 200)
    t_eval = np.arange(0, 200, 0.02)
    u0 = [1.0, 0.0, 0.0]
    sol = solve_ivp(lorenz, t_span, u0, t_eval=t_eval)
    return sol.y.T  # shape: (time, 3)

# Training function with timing
def train_and_generate(params, X_train, Y_train, scaler, predict_len, test, test_scaled):
    set_seed(42)
    
    reservoir = Reservoir(units=params['units'], sr=params['sr'], lr=params['lr'])
    readout = Ridge(ridge=params['ridge'])
    esn = reservoir >> readout

    # Timing: Training
    start_fit = time.perf_counter()
    esn = esn.fit(X_train, Y_train, warmup=100, reset=True)
    end_fit = time.perf_counter()
    fit_duration = end_fit - start_fit
    print(f"      ⏱ Training duration: {fit_duration:.4f} seconds")

    # Timing: Prediction
    start_pred = time.perf_counter()
    last_input = X_train[-1].reshape(1, -1)
    outputs = []
    step_times = []
    for step in range(predict_len):
        start_time = time.perf_counter()
        pred = esn.run(last_input)
        end_time = time.perf_counter()

        duration = end_time - start_time
        step_times.append(duration)

        print(f"Step {step}: {duration * 1e3:.6f} ms")

        outputs.append(pred.ravel())
        last_input = pred
    
    step_times_np = np.array(step_times)
    avg_time = step_times_np.mean()
    min_time = step_times_np.min()
    max_time = step_times_np.max()
    min_idx = step_times_np.argmin()
    max_idx = step_times_np.argmax()

    print(f"\nAverage prediction time per step: {avg_time * 1e3:.6f} ms")
    print(f"Minimum prediction time per step: {min_time * 1e3:.6f} ms at step {min_idx}")
    print(f"Maximum prediction time per step: {max_time * 1e3:.6f} ms at step {max_idx}")

    end_pred = time.perf_counter()
    pred_duration = end_pred - start_pred
    print(f"      ⏱ Prediction duration: {pred_duration:.4f} seconds")

    output_scaled = np.array(outputs)
    output = scaler.inverse_transform(output_scaled)

    mse = mean_squared_error(test, output)
    mse_scaled = mean_squared_error(test_scaled, output_scaled)

    return mse, mse_scaled, output, output_scaled

# Prepare data
data = generate_lorenz_data()
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

# Parameter grid
param_grid = {
    'sr': [1.5],
    'lr': [0.2],
    'units': [2200],
    'ridge': [1e-2]
}

# Search for best parameters
best_mse = float('inf')
best_mse_scaled = float('inf')
best_params = None
best_output = None
best_output_scaled = None
best_idx = None

total_start = time.perf_counter()

for idx, (sr, lr, units, ridge) in enumerate(product(param_grid['sr'], param_grid['lr'], param_grid['units'], param_grid['ridge'])):
    params = {'sr': sr, 'lr': lr, 'units': units, 'ridge': ridge}
    print(f"\n[{idx}] Testing parameters: {params}")
    
    start_model = time.perf_counter()
    mse, mse_scaled, output, output_scaled = train_and_generate(params, X_scaled, Y_scaled, scaler, predict_len, test, test_scaled)
    end_model = time.perf_counter()
    model_duration = end_model - start_model
    
    print(f"   --> MSE: {mse:.6f} | MSE_scaled: {mse_scaled:.6f}")
    print(f"   ⏱ Total time for this model: {model_duration:.2f} seconds")

    if mse < best_mse:
        print("   ✅ New best model found!")
        best_mse = mse
        best_mse_scaled = mse_scaled
        best_params = params
        best_output = output
        best_output_scaled = output_scaled
        best_idx = idx

total_end = time.perf_counter()
total_runtime = total_end - total_start
print(f"\n⏱ Total runtime for all models: {total_runtime:.2f} seconds")

# Calculate MSE per dimension
mse_x = mean_squared_error(test[:, 0], best_output[:, 0])
mse_y = mean_squared_error(test[:, 1], best_output[:, 1])
mse_z = mean_squared_error(test[:, 2], best_output[:, 2])

mse_scaled_x = mean_squared_error(test_scaled[:, 0], best_output_scaled[:, 0])
mse_scaled_y = mean_squared_error(test_scaled[:, 1], best_output_scaled[:, 1])
mse_scaled_z = mean_squared_error(test_scaled[:, 2], best_output_scaled[:, 2])

# Visualization
fig = plt.figure(figsize=(12, 8))

ax1 = fig.add_subplot(221)
ax1.plot(best_output[:, 0], label="predicted x")
ax1.plot(test[:, 0], '--', label="true x")
ax1.set_title(f"x(t)\nMSE unscaled: {mse_x:.6f} | scaled: {mse_scaled_x:.6f}")
ax1.legend()

ax2 = fig.add_subplot(222)
ax2.plot(best_output[:, 1], label="predicted y")
ax2.plot(test[:, 1], '--', label="true y")
ax2.set_title(f"y(t)\nMSE unscaled: {mse_y:.6f} | scaled: {mse_scaled_y:.6f}")
ax2.legend()

ax3 = fig.add_subplot(223)
ax3.plot(best_output[:, 2], label="predicted z")
ax3.plot(test[:, 2], '--', label="true z")
ax3.set_title(f"z(t)\nMSE unscaled: {mse_z:.6f} | scaled: {mse_scaled_z:.6f}")
ax3.legend()

ax4 = fig.add_subplot(224, projection='3d')
ax4.plot(*best_output.T, label="predicted")
ax4.plot(*test.T, linestyle="--", label="true")
ax4.set_title(
    f"Lorenz attractor\nMSE unscaled (total): {best_mse:.6f}\nMSE scaled (total): {best_mse_scaled:.6f}"
)
ax4.legend()

plt.suptitle(f"Best Hyperparameters: sr={best_params['sr']}, lr={best_params['lr']}, "
             f"units={best_params['units']}, ridge={best_params['ridge']}\n"
             f"Model Number: {best_idx}")
plt.tight_layout()
plt.show()