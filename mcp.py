import numpy as np


class MCP:
    def __init__(self, inputDimension, outputFunction=lambda x: x, initialWeights=False):
        self.inputDimension = inputDimension
        self.outputFunction = outputFunction

        if initialWeights:
            self.initialWeights = np.array(initialWeights)
            if self.initialWeights.shape[0] != self.inputDimension:
                raise Exception("Initial weights has wrong dimension")
        
        else:
            self.initialWeights = np.zeros(self.inputDimension)

    def train(self, X, y, learningRate=0.01):
        pass

    def predict(self, X):
        pass
