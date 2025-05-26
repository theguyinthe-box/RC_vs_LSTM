import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge as SklearnRidge
from sklearn.metrics import mean_squared_error
from reservoirpy.nodes import NVAR
from mpl_toolkits.mplot3d import Axes3D

# Lorenz system
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

# Parameters
delay = 2
order = 2
strides = 1
predict_len = 1250
shift = 300
train_len = 5000
warmup = 10

# Prepare data
data = generate_lorenz_data()
X = data[shift : shift + train_len]
Y = data[shift + 1 : shift + train_len + 1]
test = data[shift + train_len : shift + train_len + predict_len]

# Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
Y_scaled = scaler.transform(Y)
test_scaled = scaler.transform(test)

# NVAR feature generator (no training!)
nvar = NVAR(delay=delay, order=order, strides=strides, input_dim=3)

# Generate features (ignore warmup)
features = nvar.run(X_scaled)

# Train Ridge readout with NVAR features and targets
regressor = SklearnRidge(alpha=1e-6)
regressor.fit(features[warmup:], Y_scaled[warmup:])

# Autonomous prediction
last_input = X_scaled[-delay:].copy()
outputs = []

for _ in range(predict_len):
    feat = nvar.run(last_input)
    pred = regressor.predict(feat[-1].reshape(1, -1))
    outputs.append(pred.ravel())
    last_input = np.roll(last_input, -1, axis=0)
    last_input[-1] = pred.ravel()

# Inverse scale the output
output_scaled = np.array(outputs)
output = scaler.inverse_transform(output_scaled)

# MSE in the unscaled (original) space
mse_x = mean_squared_error(test[:, 0], output[:, 0])
mse_y = mean_squared_error(test[:, 1], output[:, 1])
mse_z = mean_squared_error(test[:, 2], output[:, 2])
mse_total = mean_squared_error(test, output)

# MSE in the scaled space
mse_scaled_x = mean_squared_error(test_scaled[:, 0], output_scaled[:, 0])
mse_scaled_y = mean_squared_error(test_scaled[:, 1], output_scaled[:, 1])
mse_scaled_z = mean_squared_error(test_scaled[:, 2], output_scaled[:, 2])
mse_scaled_total = mean_squared_error(test_scaled, output_scaled)

print(f"✅ NGRC total MSE: {mse_total:.6f}")
print(f"  - x MSE: {mse_x:.6f}")
print(f"  - y MSE: {mse_y:.6f}")
print(f"  - z MSE: {mse_z:.6f}")

print(f"✅ NGRC scaled total MSE: {mse_scaled_total:.6f}")
print(f"  - x scaled MSE: {mse_scaled_x:.6f}")
print(f"  - y scaled MSE: {mse_scaled_y:.6f}")
print(f"  - z scaled MSE: {mse_scaled_z:.6f}")

# Visualization
fig = plt.figure(figsize=(12, 8))

ax1 = fig.add_subplot(221)
ax1.plot(output[:, 0], label="predicted x")
ax1.plot(test[:, 0], '--', label="true x")
ax1.set_title(f"x(t)\nMSE: {mse_x:.6f} | scaled MSE: {mse_scaled_x:.6f}")
ax1.legend()

ax2 = fig.add_subplot(222)
ax2.plot(output[:, 1], label="predicted y")
ax2.plot(test[:, 1], '--', label="true y")
ax2.set_title(f"y(t)\nMSE: {mse_y:.6f} | scaled MSE: {mse_scaled_y:.6f}")
ax2.legend()

ax3 = fig.add_subplot(223)
ax3.plot(output[:, 2], label="predicted z")
ax3.plot(test[:, 2], '--', label="true z")
ax3.set_title(f"z(t)\nMSE: {mse_z:.6f} | scaled MSE: {mse_scaled_z:.6f}")
ax3.legend()

ax4 = fig.add_subplot(224, projection='3d')
ax4.plot(*output.T, label="predicted")
ax4.plot(*test.T, linestyle="--", label="true")
ax4.set_title(f"Lorenz Attractor\nTotal MSE: {mse_total:.6f}\nScaled Total MSE: {mse_scaled_total:.6f}")
ax4.legend()

plt.suptitle("ReservoirPy NGRC")
plt.tight_layout()
plt.show()