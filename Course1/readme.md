# Neural Network Representation

  - Input Layer ( a0 = X )
  - Hidden Layer ( a1 = hidden layer1, a2 = hidden layer2, .... )  
  - Output Layer ( an = output layer )
  - Example: input layer size 3 (x1, x2, x3), 1 hidden layer ( 4 nodes ), 1 output layer
    - 2 layer NN - Input layer is not counted as official layer
    - for hidden layer: w1 = (4,3), b1 = (4,1)
    - for output layer: w2 = (1,4), b2 = (1,1)
  
# Computing Neural Network Output

  - Simple Logistic Regression:
    - Output function: Sigmoid
    - Input variables: x1, x2, x3
    - Output = a = Sigmoid(z)
    - z = w(t)*x+b
  - Neural Network:
    - 1 hidden layer ( t = tranpose )
      - First node: z11 = w11t*x + b11, a11 = sigmoid(z11)
      - Second node: z12 = w12t*x + b12, a12 = sigmoid(z12)
      - ...
    - Z1 = [ [--w11t--], [--w12--], [--w13--] * [ X1, X2, X3] + [b11, b12, b13, b14]
    - Z1 = W1t*X + b1
    - A1 = sigmoid(Z1)
    - Z2 = W2t*A1 + b2
    - A2 = sigmoid(Z2)
    
# Vectorizing across multiple Examples

  - For i = 1 to m:
    - z1i = W1*xi + b1
    - a1i = sigmoid(z1i)
    - z2i = W2*a1i + b2
    - a2i = sigmoid(z2i)
  - X = [ X1, X2,... Xm ] - Stacking the X vectors horizontally
  - W = [ w1t, w2t,... wmt ] - Stacking the X vectors horizontally
  - A = [ A1, A2,... Am ] - Stacking the X vectors horizontally
    - A[11] = Activation of first hidden on first training example
    - A[12] = Activation of first hidden on second training example
    - A[21] = Activation of second hidden on first training example
      - horizontally for different training example
      - vertically for different hidden layers
 
 # Activation Functions
 
  - sigmoid - a(z) = 1/(1+e^-z) - values lie in the range (0 -1]
  - tanh - g(z) = tanh(z) = (e^z - e^-z)/(e^z + e^-z) - values lie in the range (-1,1) 
    - Superior compared to sigmoid except when output is 0-1 - binary classification
    - effect of centering the data - mean is 0
  - Sometimes tanh can be used in hidden layer & sigmoid in output layer
  - One problem with sigmoid, tanh -  when Z is very large or very small then the gradient is almost 0.
  - relu - a(z) = max(0,z)
    - non-differentiable at z=0 in theory but works in practice
    - derivative is 0 when z < 0
  - leaky relu - a(z) = z if z >0 else 0.01*z
    - derivative is non-zero if z < 0
  - Using relu the model learns better because of slope is not small.

# Why do you need non-linear activation functions?

  - 
