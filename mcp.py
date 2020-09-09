import numpy as np


def addBiasColumn(X):
    return np.column_stack((np.ones(X.shape[0]), X))


class MCP:
    def __init__(self, inputDimension, outputFunction=lambda x: x, initialWeights=False):
        self.inputDimension = inputDimension
        self.weightsDimension = inputDimension + 1
        self.outputFunction = outputFunction

        if initialWeights:
            self.outputWeights = np.array(initialWeights)
            if self.outputWeights.shape[0] != self.weightsDimension:
                raise Exception("Initial weights has wrong dimension")

        else:
            self.outputWeights = np.zeros(self.weightsDimension)

    def train(self, X, y, learningRate=0.01, maxIterations=1, tolerance=1e-9):
        for _ in range(maxIterations):
            iterationError = 0
            for index, row in enumerate(X):
                augmentedRow = addBiasColumn(
                    row.reshape((1, self.inputDimension))
                )
                yApprox = self.predict(augmentedRow, True)[0]

                error = y[index]-yApprox
                iterationError += error ** 2
                self.outputWeights = self.outputWeights + learningRate*error*augmentedRow

            iterationError /= X.shape[0]
            if iterationError <= tolerance:
                break
        pass

    def predict(self, X, hasBiasColumn=False):
        if hasBiasColumn:
            return self.outputWeights.dot(X.T)
        else:
            return self.outputWeights.dot(addBiasColumn(X).T)
