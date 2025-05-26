import numpy as np
import matplotlib.pyplot as plt
from reservoirpy.nodes import Reservoir, Ridge
from reservoirpy import set_seed, verbosity

# Reproduzierbarkeit und weniger Log-Ausgaben
set_seed(42)
verbosity(0)

# 1. ESN-Modell definieren
reservoir = Reservoir(100, lr=0.5, sr=0.9)
readout = Ridge(ridge=1e-7)
esn_model = reservoir >> readout

# 2. Datengenerierung: Sinuskurve
X = np.sin(np.linspace(0, 6 * np.pi, 100)).reshape(-1, 1)

# 3. Trainings- & Ziel-Daten (Ein-Schritt-Vorhersage)
X_train = X[:50]
Y_train = X[1:51]

# 4. Training des Modells (inkl. Reset und Warmup)
esn_model = esn_model.fit(X_train, Y_train, warmup=10, reset=True)

# 5. Testphase: Vorhersage auf unbekannten Daten
X_test = X[50:-1]
Y_true = X[51:]
Y_pred = esn_model.run(X_test)

# 6. Visualisierung
plt.figure(figsize=(10, 3))
plt.title("Sinus-Vorhersage mit ESN")
plt.xlabel("$t$")
plt.ylabel(r"$\sin(t)$")
plt.plot(Y_pred, label=r"Vorhersage: $\sin(t+1)$", color="blue")
plt.plot(Y_true, label=r"Wahr: $\sin(t+1)$", color="red", linestyle="dashed")
plt.legend()
plt.tight_layout()
plt.show()