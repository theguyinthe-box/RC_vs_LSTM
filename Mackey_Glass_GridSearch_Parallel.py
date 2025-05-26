import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from reservoirpy.nodes import Reservoir, Ridge
from reservoirpy import set_seed
from sklearn.metrics import mean_squared_error
from itertools import product
from joblib import Parallel, delayed

# Mackey-Glass generator
def mackey_glass(beta=0.2, gamma=0.1, n=10, tau=17, length=7000, dt=1.0):
    from collections import deque
    history = deque([1.2] * tau, maxlen=tau)
    x = []
    for _ in range(length):
        x_tau = history[0]
        x_t = history[-1]
        dx = beta * x_tau / (1 + x_tau**n) - gamma * x_t
        x_t1 = x_t + dx * dt
        history.append(x_t1)
        x.append(x_t1)
    return np.array(x)

# Generate data
series = mackey_glass()
series = series.reshape(-1, 1)

# Prepare data
shift = 0
train_len = 5000
predict_len = 1250

X = series[shift:shift+train_len]
Y = series[shift+1:shift+train_len+1]
test = series[shift+train_len:shift+train_len+predict_len]

# Scale data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
Y_scaled = scaler.transform(Y)

# Define Grid Search parameter space
"""param_grid = {
    'sr': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0],
    'lr': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    'units': [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000],
    'ridge': [1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2]
}"""

# Reduced parameter space for faster execution
param_grid = {
    'sr': [0.9, 1.0, 1.1, 1.2, 1.3],
    'lr': [0.0, 0.1, 0.2, 0.3],
    'units': [300, 400, 500, 600, 700, 800],
    'ridge': [1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2]
}

# Function for training + generative prediction + MSE
def train_and_generate(params, X_train, Y_train, scaler, predict_len, test, idx):
    set_seed(42)
    reservoir = Reservoir(units=params['units'], sr=params['sr'], lr=params['lr'])
    readout = Ridge(ridge=params['ridge'])
    esn = reservoir >> readout

    esn = esn.fit(X_train, Y_train, warmup=100, reset=True)

    last_input = X_train[-1].reshape(1, -1)
    outputs = []
    for _ in range(predict_len):
        pred = esn.run(last_input)
        outputs.append(pred.ravel())
        last_input = pred

    output_scaled = np.array(outputs)
    output = scaler.inverse_transform(output_scaled)
    
    mse = mean_squared_error(test, output)
    return mse, output, params, idx

# Search for best parameters
best_mse = float('inf')
best_params = None
best_output = None
best_idx = None

# Parallel grid search
results = Parallel(n_jobs=-1)(
    delayed(train_and_generate)(
        {'sr': sr, 'lr': lr, 'units': units, 'ridge': ridge},
        X_scaled, Y_scaled, scaler, predict_len, test, idx
    )
    for idx, (sr, lr, units, ridge) in enumerate(
        product(param_grid['sr'], param_grid['lr'], param_grid['units'], param_grid['ridge'])
    )
)

# Evaluate results
for mse, output, params, idx in results:
    print(f"[{idx}] MSE: {mse:.6f} | Model params: {params}")
    if mse < best_mse:
        print(f"   ✅ New best model found!")
        best_mse = mse
        best_params = params
        best_output = output
        best_idx = idx

# Plot
plt.figure(figsize=(12, 5))
plt.plot(best_output, label='predicted')
plt.plot(test, '--', label='true')
plt.title(f"Mackey-Glass generative prediction\n"
          f"Best Hyperparameters: sr={best_params['sr']}, lr={best_params['lr']}, "
          f"units={best_params['units']}, ridge={best_params['ridge']}\n"
          f"MSE: {best_mse:.6f} | Model Number: {best_idx}")
plt.xlabel("Time")
plt.ylabel("x(t)")
plt.legend()
plt.tight_layout()
plt.show()
