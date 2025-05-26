import numpy as np
import matplotlib.pyplot as plt
import reservoirpy as rpy

from reservoirpy.nodes import Reservoir, Ridge, Input

data = Input()
reservoir = Reservoir(100, lr=0.5, sr=0.9)
readout = Ridge(ridge=1e-7)

esn_model = data >> reservoir >> readout & data >> readout

rpy.verbosity(0)  # no need to be too verbose here
rpy.set_seed(42)  # make everything reproducible!

X = np.sin(np.linspace(0, 6*np.pi, 100)).reshape(-1, 1)

X_train = X[:50]
Y_train = X[1:51]

plt.figure(figsize=(10, 3))
plt.title("A sine wave.")
plt.ylabel("$sin(t)$")
plt.xlabel("$t$")
plt.plot(X)
plt.show()