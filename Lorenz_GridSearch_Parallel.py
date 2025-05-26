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

# Lorenz-System
def lorenz(t, u, sigma=10., rho=28., beta=8/3):
    x, y, z = u
    return [
        sigma * (y - x),
        x * (rho - z) - y,
        x * y - beta * z
    ]

# Lorenz-Daten generieren
def generate_lorenz_data():
    t_span = (0, 200)
    t_eval = np.arange(0, 200, 0.02)
    u0 = [1.0, 0.0, 0.0]
    sol = solve_ivp(lorenz, t_span, u0, t_eval=t_eval)
    return sol.y.T  # shape: (time, 3)

# Trainingsfunktion
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
    mse_scaled = mean_squared_error(scaler.transform(test), output_scaled)
    return mse, mse_scaled, output, output_scaled, params, idx

# Daten vorbereiten
data = generate_lorenz_data()
shift = 300
train_len = 5000
predict_len = 1250

X = data[shift:shift+train_len]
Y = data[shift+1:shift+train_len+1]
test = data[shift+train_len:shift+train_len+predict_len]

# Skalieren
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
Y_scaled = scaler.transform(Y)
test_scaled = scaler.transform(test)

# Reduzierter Parameter-Raum
param_grid = {
    'sr': [1.5],
    'lr': [0.2],
    'units': [2200],
    'ridge': [1e-2]
}

# Parallel ausführen
results = Parallel(n_jobs=-1)(
    delayed(train_and_generate)(
        {'sr': sr, 'lr': lr, 'units': units, 'ridge': ridge}, 
        X_scaled, Y_scaled, scaler, predict_len, test, idx
    )
    for idx, (sr, lr, units, ridge) in enumerate(
        product(param_grid['sr'], param_grid['lr'], param_grid['units'], param_grid['ridge'])
    )
)

# Ergebnisse ausgeben und bestes Modell merken
best_mse = float('inf')
best_mse_scaled = float('inf')
best_params = None
best_output = None
best_output_scaled = None
best_idx = None

for mse, mse_scaled, output, output_scaled, params, idx in results:
    print(f"[{idx}] MSE: {mse:.6f} | MSE_scaled: {mse_scaled:.6f} | Model params: {params}")
    if mse < best_mse:
        best_mse = mse
        best_mse_scaled = mse_scaled
        best_params = params
        best_output = output
        best_output_scaled = output_scaled
        best_idx = idx
        print(f"   ✅ New best model found! (Model #{best_idx})")

# Nach der Suche nochmal ausführliche Ausgabe des besten Modells:
print("\n=== Bestes Modell ===")
print(f"Modellnummer: {best_idx}")
print(f"Hyperparameter: sr={best_params['sr']}, lr={best_params['lr']}, units={best_params['units']}, ridge={best_params['ridge']}")
print(f"MSE unskaliert (gesamt): {best_mse:.6f}")
print(f"MSE skaliert (gesamt): {best_mse_scaled:.6f}")

# MSE im unskalierten Raum
mse_x = mean_squared_error(test[:, 0], best_output[:, 0])
mse_y = mean_squared_error(test[:, 1], best_output[:, 1])
mse_z = mean_squared_error(test[:, 2], best_output[:, 2])

# MSE im skalierten Raum (gesamt und pro Dimension)
mse_scaled = mean_squared_error(test_scaled, best_output_scaled)
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
    f"Lorenz attractor\nMSE unscaled (total): {best_mse:.6f}\nMSE scaled (total): {mse_scaled:.6f}"
)
ax4.legend()

plt.suptitle(f"Best Hyperparameters: sr={best_params['sr']}, lr={best_params['lr']}, "
             f"units={best_params['units']}, ridge={best_params['ridge']}\n"
             f"Model Number: {best_idx}")
plt.tight_layout()
plt.show()