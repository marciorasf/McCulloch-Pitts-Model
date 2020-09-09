import numpy as np

from mcp import MCP, Perceptron, Adaline

mcp = MCP(2)

X = np.array([[1, 1], [1, 2], [1, 3], [2,1], [3, 1]])
y = np.array([1, 2, 3, 2, 3])

mcp.train(X, y, 0.1, 1000, 1e-12)
y = mcp.predict(X)
print(y)
print(mcp.outputWeights)
