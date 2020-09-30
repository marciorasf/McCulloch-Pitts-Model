# MCP - McCulloch - Pitts

## Features
- All MCP structure
  
- It's simplicity make it great for students that are having the first contact with MCP
  
- Generic implementation
 
  - Accepts the output function as parameter
  - Accepts the initial weights as parameter
  
- Offers Adaline and Perceptron implementations out of the box


## API

### Instantiate MCP

```
myMcp = MCP(inputDimension, outputFunction, initialWeights)
```
1. `myMcp` is a instance of MCP

2. **inputDimension**: the dimension of the problem. The bias term should not be considered

3. **ouputFunction** (optional): function used to transform the output. By default it is the identity function x->x

4. **initialWeights** (optional): used in case you have initial weights for the MCP. You should include the bias weight


#### You can also use Adaline or Perceptron to use them **outputFunction**:

```
myAdaline = Adaline(inputDimension, initialWeights)
myPerceptron = Perceptron(inputDimension, initialWeights)
```

### Train

```
myMcp.train(X, y, learningRate, maxIterations, tolerance):
```

1. **X**: matrix with the input data

2. **y**: array with the expected output for each input

3. **learningRate** (optional): factor that determine how big is the response for the error between the expected result and tha actual result obtained by the model

4. **maxIterations** (optional): max iterations of the training session

5. **tolerance** (optional): value the determines how small the mean squared error should be so the training session can stop

### Predict

```
yApprox = myMcp.predict(X):
```

1. `yApprox` is the result array obtained by the trained model

2. **X**: matrix with the input data
