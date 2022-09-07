import pdb
from matplotlib import pyplot as plt
import reservoirpy as respy
from reservoirpy import datasets
from reservoirpy.observables import rmse, rsquare
import numpy as np

# reservoir = respy.nodes.Reservoir(100, lr=0.5, sr=0.9)
# readout = respy.nodes.Ridge(ridge=1e-6)

# X = datasets.mackey_glass(2000)

# X = np.sin(np.linspace(0,20, 100))[:, np.newaxis]
# y = np.cos(np.linspace(0,20,100))[:, np.newaxis]
# esn = reservoir >> readout

# esn.fit(X[:500], X[1:501], warmup=100)

# predictions = esn.run(X[501:-1])

# print("RMSE:", rmse(X[502:], predictions), "R^2 score:", rsquare(X[502:], predictions))

# pdb.set_trace()

1