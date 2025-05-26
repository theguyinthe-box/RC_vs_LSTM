import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from reservoirpy.nodes import Reservoir, Ridge
from reservoirpy import set_seed

# Seed setzen
set_seed(42)

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

# RC-Modell
reservoir = Reservoir(units=500, sr=1.25, lr=0.5)
readout = Ridge(ridge=1e-6)
esn = reservoir >> readout

# Training
esn = esn.fit(X_scaled, Y_scaled, warmup=100, reset=True)

# Generative Vorhersage
last_input = X_scaled[-1].reshape(1, -1)
outputs = []

for _ in range(predict_len):
    pred = esn.run(last_input)
    outputs.append(pred.ravel())
    last_input = pred  # Feedback

output_scaled = np.array(outputs)
output = scaler.inverse_transform(output_scaled)

# Plot
plt.figure(figsize=(12, 5))
plt.plot(output, label='predicted')
plt.plot(test, '--', label='true')
plt.title("Mackey-Glass generative prediction")
plt.xlabel("Time")
plt.ylabel("x(t)")
plt.legend()
plt.tight_layout()
plt.show()