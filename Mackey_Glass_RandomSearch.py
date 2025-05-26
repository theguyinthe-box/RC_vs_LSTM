import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from reservoirpy.nodes import Reservoir, Ridge
from reservoirpy import set_seed
from sklearn.metrics import mean_squared_error
import random

# Set seed
set_seed(42)

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
series = series.reshape(-1, 1)  # Shape: (time, 1)

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

# Define hyperparameter search space for Random Search
param_grid = {
    'sr': [1.0, 1.1, 1.2, 1.3, 1.4],
    'lr': [0.3, 0.5, 0.7, 0.9],
    'units': [300, 400, 500, 600]
}

# Number of random search iterations
n_iter = 80  # For example: try 80 random combinations

best_mse = float('inf')
best_params = None
best_esn = None

# Random Search: try random combinations of hyperparameters
for _ in range(n_iter):
    # Randomly select hyperparameters
    params = {
        'sr': random.choice(param_grid['sr']),
        'lr': random.choice(param_grid['lr']),
        'units': random.choice(param_grid['units'])
    }
    
    # Create reservoir with the chosen hyperparameters
    reservoir = Reservoir(units=params['units'], sr=params['sr'], lr=params['lr'])
    readout = Ridge(ridge=1e-6)
    esn = reservoir >> readout
    
    # Train the model
    esn = esn.fit(X_scaled, Y_scaled, warmup=100, reset=True)
    
    # Predict on test data
    last_input = X_scaled[-1].reshape(1, -1)
    outputs = []
    
    for _ in range(predict_len):
        pred = esn.run(last_input)
        outputs.append(pred.ravel())
        last_input = pred  # Feedback loop

    output_scaled = np.array(outputs)
    output = scaler.inverse_transform(output_scaled)
    
    # Compute MSE
    mse = mean_squared_error(test, output)
    
    # Save parameters and model if current MSE is the best so far
    if mse < best_mse:
        best_mse = mse
        best_params = params
        best_esn = esn

# Show best hyperparameters and model performance
print("Best hyperparameters:")
print(f"sr: {best_params['sr']}")
print(f"lr: {best_params['lr']}")
print(f"units: {best_params['units']}")
print(f"Best MSE: {best_mse}")

# Generative prediction with best model
last_input = X_scaled[-1].reshape(1, -1)
outputs = []

for _ in range(predict_len):
    pred = best_esn.run(last_input)
    outputs.append(pred.ravel())
    last_input = pred

output_scaled = np.array(outputs)
output = scaler.inverse_transform(output_scaled)

# Plot the results
plt.figure(figsize=(12, 5))
plt.plot(output, label='Predicted')
plt.plot(test, '--', label='True')
plt.title(f"Mackey-Glass generative prediction\nBest Model with sr={best_params['sr']}, lr={best_params['lr']}, units={best_params['units']}")
plt.xlabel("Time")
plt.ylabel("x(t)")
plt.legend()
plt.tight_layout()
plt.show()