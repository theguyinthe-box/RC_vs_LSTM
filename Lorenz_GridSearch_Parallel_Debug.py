import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from reservoirpy.nodes import Reservoir, Ridge
from reservoirpy import set_seed
from itertools import product
from joblib import Parallel, delayed
from mpl_toolkits.mplot3d import Axes3D

# Lorenz system definition
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

# Training function
def train_and_generate(params, X_train, Y_train, scaler, predict_len, test, idx):
    set_seed(42)
    reservoir = Reservoir(units=params['units'], sr=params['sr'], lr=params['lr'])
    readout = Ridge(ridge=params['ridge'])
    esn = reservoir >> readout

    try:
        esn = esn.fit(X_train, Y_train, warmup=100, reset=True)
    except Exception as e:
        print(f"[{idx}] ⚠️ Training failed: {e}")
        return float('inf'), None, params, idx

    last_input = X_train[-1].reshape(1, -1)
    outputs = []
    for _ in range(predict_len):
        pred = esn.run(last_input)
        outputs.append(pred.ravel())
        last_input = pred

    output_scaled = np.array(outputs)
    output = scaler.inverse_transform(output_scaled)

    # ⚠️ Important check: Is the prediction degenerate (almost constant)?
    if np.all(np.std(output, axis=0) < 1e-4):
        print(f"[{idx}] ❌ Degenerate output detected (almost constant): {np.std(output, axis=0)}")
        return float('inf'), None, params, idx

    mse = mean_squared_error(test, output)
    return mse, output, params, idx

# Prepare data
data = generate_lorenz_data()
shift = 300
train_len = 5000
predict_len = 500

X = data[shift:shift+train_len]
Y = data[shift+1:shift+train_len+1]
test = data[shift+train_len:shift+train_len+predict_len]

# Scale data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
Y_scaled = scaler.transform(Y)

# Reduced parameter grid for faster execution
param_grid = {
    'sr': [1.0, 1.1, 1.2, 1.3, 1.4, 1.5],
    'lr': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
    'units': [500, 600, 700, 800, 900, 1000],
    'ridge': [1e-7, 1e-6, 1e-5]
}

# Run training in parallel
results = Parallel(n_jobs=-1)(
    delayed(train_and_generate)(
        {'sr': sr, 'lr': lr, 'units': units, 'ridge': ridge}, 
        X_scaled, Y_scaled, scaler, predict_len, test, idx
    )
    for idx, (sr, lr, units, ridge) in enumerate(
        product(param_grid['sr'], param_grid['lr'], param_grid['units'], param_grid['ridge'])
    )
)

# Find best result
best_mse = float('inf')
best_params = None
best_output = None
best_idx = None

for mse, output, params, idx in results:
    print(f"[{idx}] MSE: {mse:.6f} | Model params: {params}")
    if mse < best_mse:
        print(f"   ✅ New best model found!")
        best_mse = mse
        best_params = params
        best_output = output
        best_idx = idx

# Visualize results
mse_x = mean_squared_error(test[:, 0], best_output[:, 0])
mse_y = mean_squared_error(test[:, 1], best_output[:, 1])
mse_z = mean_squared_error(test[:, 2], best_output[:, 2])

fig = plt.figure(figsize=(12, 8))

ax1 = fig.add_subplot(221)
ax1.plot(best_output[:, 0], label="predicted x")
ax1.plot(test[:, 0], '--', label="true x")
ax1.set_title(f"x(t) — MSE: {mse_x:.6f}")
ax1.legend()

ax2 = fig.add_subplot(222)
ax2.plot(best_output[:, 1], label="predicted y")
ax2.plot(test[:, 1], '--', label="true y")
ax2.set_title(f"y(t) — MSE: {mse_y:.6f}")
ax2.legend()

ax3 = fig.add_subplot(223)
ax3.plot(best_output[:, 2], label="predicted z")
ax3.plot(test[:, 2], '--', label="true z")
ax3.set_title(f"z(t) — MSE: {mse_z:.6f}")
ax3.legend()

ax4 = fig.add_subplot(224, projection='3d')
ax4.plot(*best_output.T, label="predicted trajectory")
ax4.plot(*test.T, linestyle="--", label="true trajectory")
ax4.set_title(f"Lorenz attractor\nTotal MSE: {best_mse:.6f}")
ax4.legend()

plt.suptitle(f"Best Hyperparameters: sr={best_params['sr']}, lr={best_params['lr']}, "
             f"units={best_params['units']}, ridge={best_params['ridge']}\n"
             f"Model Number: {best_idx}")
plt.tight_layout()
plt.show()