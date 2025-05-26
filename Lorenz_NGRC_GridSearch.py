import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge as SklearnRidge
from sklearn.metrics import mean_squared_error
from reservoirpy.nodes import NVAR
from mpl_toolkits.mplot3d import Axes3D
from itertools import product

def lorenz(t, u, sigma=10., rho=28., beta=8/3):
    x, y, z = u
    return [
        sigma * (y - x),
        x * (rho - z) - y,
        x * y - beta * z
    ]

def generate_lorenz_data():
    t_span = (0, 200)
    t_eval = np.arange(0, 200, 0.02)
    u0 = [1.0, 0.0, 0.0]
    sol = solve_ivp(lorenz, t_span, u0, t_eval=t_eval)
    return sol.y.T

predict_len = 1250
shift = 300
train_len = 5000
warmup = 10

data = generate_lorenz_data()
X = data[shift : shift + train_len]
Y = data[shift + 1 : shift + train_len + 1]
test = data[shift + train_len : shift + train_len + predict_len]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
Y_scaled = scaler.transform(Y)
test_scaled = scaler.transform(test)

param_grid = {
    "delay": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "order": [2, 3],
    "strides": [1, 2, 3],
    "alpha": [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
}

best_mse = float("inf")
best_params = None
best_output = None
best_idx = None

print("Starte Grid Search...")

for idx, (delay, order, strides, alpha) in enumerate(
    product(param_grid["delay"], param_grid["order"], param_grid["strides"], param_grid["alpha"])
):
    try:
        nvar = NVAR(delay=delay, order=order, strides=strides, input_dim=3)
        features = nvar.run(X_scaled)

        regressor = SklearnRidge(alpha=alpha)
        regressor.fit(features[warmup:], Y_scaled[warmup:])

        last_input = X_scaled[-delay:].copy()
        outputs = []

        for _ in range(predict_len):
            feat = nvar.run(last_input)
            pred = regressor.predict(feat[-1].reshape(1, -1))
            outputs.append(pred.ravel())
            last_input = np.roll(last_input, -1, axis=0)
            last_input[-1] = pred.ravel()

        output_scaled = np.array(outputs)
        output = scaler.inverse_transform(output_scaled)

        mse = mean_squared_error(test, output)

        print(f"[{idx}] Params: delay={delay}, order={order}, strides={strides}, alpha={alpha:.0e} --> MSE: {mse:.6f}")

        if mse < best_mse:
            best_mse = mse
            best_params = (delay, order, strides, alpha)
            best_output = output
            best_idx = idx
            print(f"   ✅ New best model found! (Model #{best_idx})")

    except Exception as e:
        print(f"Fehler bei Parametern delay={delay}, order={order}, strides={strides}, alpha={alpha:.0e}: {e}")

print("\n=== Beste Parameter gefunden ===")
print(f"Modellnummer: {best_idx}")
print(f"Delay = {best_params[0]}, Order = {best_params[1]}, Strides = {best_params[2]}, Alpha = {best_params[3]:.0e}")
print(f"Mit MSE = {best_mse:.6f}")

# Skalierten Output und Testdaten vorbereiten
output_scaled = scaler.transform(best_output)
test_scaled_local = scaler.transform(test)

# MSE unskaliert für jede Dimension
mse_x = mean_squared_error(test[:, 0], best_output[:, 0])
mse_y = mean_squared_error(test[:, 1], best_output[:, 1])
mse_z = mean_squared_error(test[:, 2], best_output[:, 2])

# MSE skaliert für jede Dimension und gesamt
mse_scaled_x = mean_squared_error(test_scaled_local[:, 0], output_scaled[:, 0])
mse_scaled_y = mean_squared_error(test_scaled_local[:, 1], output_scaled[:, 1])
mse_scaled_z = mean_squared_error(test_scaled_local[:, 2], output_scaled[:, 2])
mse_scaled_total = mean_squared_error(test_scaled_local, output_scaled)

delay, order, strides, alpha = best_params

# Visualisierung
fig = plt.figure(figsize=(12, 8))

ax1 = fig.add_subplot(221)
ax1.plot(best_output[:, 0], label="pred x")
ax1.plot(test[:, 0], '--', label="true x")
ax1.set_title(f"x(t)\nMSE unscaled: {mse_x:.6f} | MSE scaled: {mse_scaled_x:.6f}")
ax1.legend()

ax2 = fig.add_subplot(222)
ax2.plot(best_output[:, 1], label="pred y")
ax2.plot(test[:, 1], '--', label="true y")
ax2.set_title(f"y(t)\nMSE unscaled: {mse_y:.6f} | MSE scaled: {mse_scaled_y:.6f}")
ax2.legend()

ax3 = fig.add_subplot(223)
ax3.plot(best_output[:, 2], label="pred z")
ax3.plot(test[:, 2], '--', label="true z")
ax3.set_title(f"z(t)\nMSE unscaled: {mse_z:.6f} | MSE scaled: {mse_scaled_z:.6f}")
ax3.legend()

ax4 = fig.add_subplot(224, projection='3d')
ax4.plot(*best_output.T, label="predicted")
ax4.plot(*test.T, linestyle="--", label="true")
ax4.set_title(f"Lorenz Attractor\nMSE unscaled total: {best_mse:.6f}\nMSE scaled total: {mse_scaled_total:.6f}")
ax4.legend()

plt.suptitle(
    f"ReservoirPy NGRC Grid Search Best Result\n"
    f"Params: delay={delay}, order={order}, strides={strides}, alpha={alpha:.0e}"
)
plt.tight_layout()
plt.show()