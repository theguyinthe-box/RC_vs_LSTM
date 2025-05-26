import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from reservoirpy.nodes import Reservoir, Ridge
from reservoirpy import set_seed
from sklearn.metrics import mean_squared_error
from itertools import product

# Mackey-Glass-Generator
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

# Daten erzeugen
series = mackey_glass()
series = series.reshape(-1, 1)  # Form: (zeit, 1)

# Daten vorbereiten
shift = 0
train_len = 5000
predict_len = 1250

X = series[shift:shift+train_len]
Y = series[shift+1:shift+train_len+1]
test = series[shift+train_len:shift+train_len+predict_len]

# Skalieren
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
Y_scaled = scaler.transform(Y)

# Grid Search Parameter-Raum definieren
param_grid = {
    'sr': [1.0, 1.1, 1.2, 1.3, 1.4, 1.5],
    'lr': [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    'units': [300, 400, 500, 600, 700, 800, 900, 1000]
}

# Reduzierter Parameter-Raum für schnellere Ausführung
"""param_grid = {
    'sr': [1.1, 1.2, 1.3],
    'lr': [0.5, 0.6, 0.7],
    'units': [600, 700, 800, 900, 1000]
}"""

# Funktion für Training + generative Vorhersage + MSE
def train_and_generate(params, X_train, Y_train, scaler, predict_len, test):

    # Seed setzen
    set_seed(42)

    reservoir = Reservoir(units=params['units'], sr=params['sr'], lr=params['lr'])
    readout = Ridge(ridge=1e-6)
    esn = reservoir >> readout

    esn = esn.fit(X_train, Y_train, warmup=100, reset=True)

    # Generative Vorhersage
    last_input = X_train[-1].reshape(1, -1)
    outputs = []
    for _ in range(predict_len):
        pred = esn.run(last_input)
        outputs.append(pred.ravel())
        last_input = pred

    output_scaled = np.array(outputs)
    output = scaler.inverse_transform(output_scaled)
    
    mse = mean_squared_error(test, output)
    return mse, output

# Beste Parameter suchen (basierend auf generativem MSE)
best_mse = float('inf')
best_params = None
best_output = None
best_idx = None  # Hinzufügen der Modellnummer

for idx, (sr, lr, units) in enumerate(product(param_grid['sr'], param_grid['lr'], param_grid['units'])):
    params = {'sr': sr, 'lr': lr, 'units': units}
    print(f"[{idx}] Testing params: sr={sr}, lr={lr}, units={units}")
    
    mse, output = train_and_generate(params, X_scaled, Y_scaled, scaler, predict_len, test)
    
    print(f"   --> MSE: {mse:.6f}")
    
    if mse < best_mse:
        print(f"   ✅ New best model found!")
        best_mse = mse
        best_params = params
        best_output = output
        best_idx = idx  # Modellnummer speichern

# Plot
plt.figure(figsize=(12, 5))
plt.plot(best_output, label='predicted')
plt.plot(test, '--', label='true')
plt.title(f"Mackey-Glass generative prediction\n"
          f"Best Hyperparameters: sr={best_params['sr']}, lr={best_params['lr']}, units={best_params['units']}\n"
          f"MSE: {best_mse:.6f} | Model Number: {best_idx}")
plt.xlabel("Time")
plt.ylabel("x(t)")
plt.legend()
plt.tight_layout()
plt.show()