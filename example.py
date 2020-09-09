import numpy as np

from mcp import MCP

mcp = MCP(2)

X = np.array([[1, 1], [1, 2], [1, 3]])
y = np.array([2, 3, 4])

mcp.train(X, y, 0.01, 100)
y = mcp.predict(X)
print(y)
