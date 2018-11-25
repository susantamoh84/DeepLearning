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

  - Without non-linear activation functions, the model is as good as no hidden layers
  - Model becomes similar to like a logistic regression

# Derivatives of activation function

  - sigmoid - a(z) = 1/(1+e^-z)
    - da/dz = 1/(1+e^-z) ( 1- 1/(1+e^-z)) = a(1-a)
      - when z is large(10)= a(z) = 1; da/dz = 0
      - when z is small(-10)= a(z) = 0; da/dz = 0
      - when z is 0 = a(z) = 0.5; da/dz = 0.5*0.5 = 0.25
  - tanh - g(z) = (e^z - e^-z)/(e^z + e^-z)
    - dg/dz = 1-g(z)^2
      - when z is large (10) = g(z) = 1; dg/dz=0
      - when z is small(-10) = g(z) = -1; dg/dz=0
      - when z is 0 = g(z) = 0; dg/dz=1
  - relu = g(z) = max(0,z)
    - dg/dz =
      - 0 if z<0
      - 1 if z>=0 ( in practice ) [ 1 if z>0 and undefined if z=0 ( theoretically ) ]
  - Leaky relu = g(z) = max(0.01z,z)
    - dg/dz=
      - 0.01 if z<0
      - 1 if z>=0 ( in practice )      

# Gradient Descent For Neural Networks Implementation

  - For one layer NN, parameters are w1, b1, w2, b2
    - w1 size = (n1, n0)
    - b1 size = (n1, 1)
    - w2 size = (n2, n1)
    - b2 size = (n2, 1)
    - for single hidden layer, n2=1, n0=nx, n1=number of nodes in hidden layer
    
  - Cost Function:
    - J(w1,b1,w2,b2) = (1/m)*(sum(Loss(yhat,y))) - yhat = a2
    
  - Gradient Descent:
    - Initialize parameters w, b to zero ( can be different )
    - Loop of Gradient Descent:
      - Compute the predictions - Forward propagation
      - Compute the derivatives - Backward propagation
      - Update the parameters w,b
        - w1 = w1 - alpha*dw1/dz
        - b1 = b1 - alpha*db1/dz
        - w2 = w2 - alpha*dw2/dz
        - b2 = b2 - alpha*db2/dz  
        
    - Vectorized Format:
      - Forward Propagation:
        - Z1 = W1*X + B1
        - A1 = g(Z1)
        - Z2 = W2*A1 + B2
        - A2 = g(Z2)
      - Backward Propagation:
        - dZ2 = A2 - Y
        - dW2 = (1/m)*dZ2*A1t
        - dB2 = (1/m)np.sum( dZ2, axis=1, keepDims=True)
        - dZ1 = (W2t*dZ2) ** dA1   ** - elementwise product
        - dW1 = (1/m)*dZ1*Xt
        - dB1 = (1/m)np.sum( dZ1, axis=1, keepDims= True)
        
# Random Initializations

  - Logistic regression initialize weights to zero
  - Random initalization for neural networks.
    - Problem with zero initalization:
      - a11 = a12 as weights are same [ assuming b=0 ]
      - dz1 = dz2 by symmetry
      - dw has same values in the rows => w has the same values in rows
      - a11 = a12 will be always same as weight vectors are same
      - in a large NN, all the hidden units are computing the same function
  - Random weight initialization: 
    - W1 = np.random.randn((2,2))*0.01 - why 0.01 - we want the initalized weights to be very small.
    - B1 = np.zeros((2,1))
      
      
# Forward-Propagation In a Deep Network

  - Example NN - 2 input layers, 5 hidden layer 1, 5 hidden layer 2, 3 hidden layer 3, 1 output layer
    - Z1 = W1 * X + b1
    - A1 = g(Z1)
    - Z2 = W2 * A1 + b2
    - A2 = g(Z2)
    - Z3 = W3 * A2 + b3
    - A3 = g(Z3)
    - Z4 = W4 * A3 + b4
    - A4 = g(Z4)
  - Generic Function:
    - Forward Propagation: 
      - Z[l] = W[l] * A[l-1] + b[l]
      - A[l] = g[l] Z[l]
      
# Backward-propagation in a Deep Network

  - dZ[l] = dA[l] * g[l] Z[l]
  - dW[l] = (1/m) dZ[l]* A[l-1]t
  - db[l] = (1/m) np.sum ( dZ[l] , axis=1, keepdims=True )
  - dA[l-1] = W[l]t * dZ[l]
      
# Hyper-parameters

  - learning rate alpha
  - #iterations
  - # hidden layers L
  - # hidden units n[1], n[2]... n[l]
  - Choice of activation function
  
![Fwd/Bkwd propagation](https://github.com/susantamoh84/DeepLearning/blob/master/forward-backward%20prop.GIF)
