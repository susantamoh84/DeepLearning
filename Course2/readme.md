# Improving Deep Neural Networks: Hyperparameter tuning, Regularization and Optimization

![Applied ML](https://github.com/susantamoh84/DeepLearning/blob/master/Course2/applied%20nn.GIF)

# Bias-Variance

![Bias-variance](https://github.com/susantamoh84/DeepLearning/blob/master/Course2/bias-variance.GIF)

# Example of Bias-Variance

  - For Example in the cat classification:
    - Case 1:
      - Training Set error - 1%
      - Dev Set Error - 11%
        - High variance - overfitting the training data
    - Case 2:
      - Training Set Error - 15%
      - Dev Set Error - 16%
      - Human Error - 0%
        - High Bias - underfitting the data
    - Case 3:
      - Training Set Error - 15%
      - Dev Set Error - 30%
        - High Bias +  High Variance - Underfitting the data and worse in Dev set.
    - Case 4:
      - Training Set Error - 0.5%
      - Dev Set Error - 1%
        - low bias + low variance

  - Bias-Variance depends on the Human Error or Optimal Error or Bayesian Error
  
# L2 Regularization
  
  ![L2 Regularization](https://github.com/susantamoh84/DeepLearning/blob/master/Course2/L2%20Regularization.GIF)
  - Weight decay
    - Regularization technique (L2) which results in gradient descent shriking the weights on every iteration

# Dropout Regularization
  
  - Randomly drop-out a portion of the units in each hidden layer  
  ![Dropout](https://github.com/susantamoh84/DeepLearning/blob/master/Course2/dropout.GIF)
  - During drop-out randomly sampled neurons < p ( keep_prob ) are deleted from each hidden layer
  - During testing time, the neurons are not deleted and hence the activations are scaled by a factor of 1/p.
  - In inverted dropouts the output of neurons are scaled by a factor of 1/p after deleting the neurons.
    - No scaling is required in the testing phase.

# Data Augumentation - Regularization

  - Image flip horizontally/vertically
  - Rotate
  - Zoom-in/out
  - Add distortion
  
# Early Stopping

  - Early stopping on train data helps overfitting
  
# Setting Up Optimization Problem:

  - Normalizaing the training Sets
    - subtracting mean
    - normalize variance
    - without normalization the gradient descent requires a very small learning rate to converge
    
  - Vanishing/Exploding gradients in very deep networks
    - activation value decrese/increase exponentially with increasing in number of layers
    - with similar argument it can be shown the gradients/derivatives decrease/increase with increasing in number of layers
    - Because of above 2 problems the training becomes extremly difficult and will take very very long time.
    
  - Better/Careful Choice of Random Weights to solve above problem
    - Z= W1*X1+W2*X2....+Wn*Xn
    - Large N -> smaller Wi
    - Var(W) = 2/n
    - W[l] = np.random.randn(shape) * np.sqrt( 2/ n[l-1] ) - This influences activation function using relu and the activations remain within a range.
    - for tanh activation: W[l] = np.random.randn(shape) * np.sqrt( 1/ n[l-1] ) - Xavier initialization

# Numerical approximations of gradients:

  - gradient = ( f(theta+episilon) - f(theta-episilon) ) / 2*episilon
  - this two-sided approach is much better than one sided difference
  
# Gradient Checking:

  - Verify the gradients
  - Take W[1], b[1], ... W[l], b[l] and reshape into a big vector theta.
  - Take dW[1], db[1], ... dW[l], db[l] and reshape into a big vector d-theta.
  - for each i:
    - d-theta[i] = ( J(theta1, theta2... thetai+episilon, ...) - J(theta1, theta2... thetai-episilon, ...) ) / 2*episilon
    - d-theta[i] = dJ/d-thetai
    - take episilon = 10^-7
    - check (||d-theta(approx) - d-theta||2) / (||d-theta(approx)||2 + ||d-theta||2)
      - if 10^-7 - great
      - if 10^-5 - ok
      - if 10^-3 - worry      

# Gradient checking implementations

  - Don't use in training - only to debug
  - if algorithim fails grad check, look at components to find bug
  - Remember regularization
  - Doesn't work with dropout.

    
# Batch vs mini-batch gradient descent

  - if m=5,000,000 the batch training becomes very slow.
  - split-up the training set into smaller batches ( mini-batch size = 1000 )
  - for t=1 ,... 5000
    - Forward Prop on X[t], Y[t]
      - X[1] = W[1] * X[t] + b[1]
      -....
      - A[l] = g[l] Z[l]
    - Compute cost function J = (1/1000) * [ sum(loss(yhat, y)) + lambda/(2*1000) sum(square(||W[l]||)) ]
    - Backprop to compute grads for J[t] using ( X[t], Y[t] )
    - Update weights, constants
    - "1 epoch" - single pass through training set.
  - Total 5000 epochs before training all data.
  - This runs much faster than batch gradient descent.
  - when mini-batch size = 1, its called stochastic gradient descent - every example is its own mini-batch
  - when mini-batch size = m, its batch gradient descent
  - typical mini-batch sizes: 64, 128, 256, 512, 1024 ( rare )
  - make sure mini-batch ( X[t], Y[t] ) fits in CPU/GPU memory.
  
# Exponentially Weighted (Moving) Averages

  - Vt = beta * Vt-1 + (1-beta) * theta-t
  - Vt as approximately avg. over ~ 1/(1-beta) days tempr
    - beta = 0.9 : ~10 days tempr avg.
    - beta = 0.98 : ~50 days tempr avg.
    - beta = 0.5 : ~2 days tempr avg.
    
  - V100 = 0.1*theta-100  + 0.1*0.9*theta-99 + 0.1*0.9^2*theta-98 + 0.1*0.9^3*theta-97 + ....
  - V100 = sum(0.1*0.9^i*theta-(100-i))
  - in 10 days decay is 1/e ~ 1/3
  
# Bias Correction

  - Vt = beta * Vt-1 + (1-beta) * theta-t
  - bias correction: Vt / (1 - beta^t) - very important in the early phase
  
# Gradient descent with momentum ( Speed up gradient descent )

  - on iteration t:
    - compute dW, db on the current mini-batch
    - Vdw = beta*Vdw + (1-beta)*dW - smoothen the dW ( sometimes 1-beta term is omitted - Vdw = beta*Vdw - not very intutive , alpha is different)
    - Vdb = beta*Vdb + (1-beta)*db - smoothen the db
    - W = W - alpha*Vdw
    - b = b - alpha*Vdb
    - practically bias-correction is not required.
  - 2 hyper-parameters: alpha, beta, in practice beta=0.9
  
# RMSProp ( Speed up gradient descent )

  - on iteration t:
    - compute dW, db on the current mini-batch
    - Sdw = beta*Sdw + (1-beta)*SW^2 
    - Sdb = beta*Sdb + (1-beta)*db^2
    - W = W - alpha*dw/sqrt(Sdw + epsilon) - epsilon is added to avoid divide by 0. 
    - b = b - alpha*db/sqrt(Sdb + epsilon)

# Adam optimization

  - on iteration t:
    - compute dW, db on the current mini-batch
    - Vdw = beta*Vdw + (1-beta1)*dW
    - Vdb = beta*Vdb + (1-beta1)*db
    - Sdw = beta*Sdw + (1-beta2)*dW^2 
    - Sdb = beta*Sdb + (1-beta2)*db^2
    - Vcorrected-dw = Vdw/(1-beta1^t)
    - Vcorrected-db = Vdb/(1-beta1^t)    
    - Scorrected-dw = Sdw/(1-beta2^t)
    - Scorrected-db = Sdb/(1-beta2^t)       
    - W = W - alpha*Vcorrected-dw/sqrt(Scorrected-dw + epsilon) 
    - b = b - alpha*Vcorrected-db/sqrt(Scorrected-db + epsilon)
    
  - Hyper-parameters tuning 
    - alpha - needs to be tuned
    - beta1 - 0.9   (dw)
    - beta2 - 0.999 (dw^2)
    - epsilon - 10^-8
    
  - Adam - Adaptive moment estimation
  
# Learning rate decay

  - slowly reduce alpha - as learning converges it should take slower learning steps
  - alpha = 1/(1 + decay_rate * epoch_run) * alpha-0
  - alpha = 0.95 ^ epoch_run * alpha-0 - exponentially decay
  - alpha = K/sqrt(apoch_run) * alpha-0 or K/sqrt(t) * alpha-0 
  - manual decay
  
# Local Optima

  - saddle point - most chances of getting these - not a local optima
    - derivative is 0
  - the more number of parameters, the more chances of getting saddle points
  - Unlikely to get stuck in a bad locl optima
  - plateaus can make learning slow
    - adam, momentum will help

# Tuning Hyper-parameters

  - Don't use grid, use random choices
  - coarse sampling
  - finer sampling in a subset of hyper-parameters
  
# Picking hyper-paramters at random

  - uniform scale sampling e.g. for alpha from 0.0001 to 1 -  this will sample more points from 0.1 to 1
  - log scale sampling e.g. for alpha from 0.0001 to 1 - equally sample values from 0.0001, 0.001, 0.01, 0.1, 1
  - for exponentially weighted average
    - beta = 0.9, 0.99, 0.999....
    - 1-beta = 0.1, 0.01, 0.001 ...
      - r in [-1, .. -3] <- sampling in log scale
      - 1-beta = 10^r
      - beta = 1 - 10^r

# Batch Normalization for speeding learning

  - Normalize the inputs
  - Normalize the Z values
    - mu = 1/m * sum( Zi )
    - sigma^2 = 1/m * sum ( (Zi - mu)^2 )
    - Znorm[i] = (Zi - mu)/(sigma^2 + epsilon)
    - Z~[i] = gamma( Znorm[i] ) * Znorm[i] + beta
    
  - It works on one minibatch at a time
    
  - Implement gradient descent
    - for t=1 to numMiniBatches
      - compute forward propagation on X[t]
        - In each hidden layer use BN ( batch normalization ) to replace Z[l] with Z~[l]
      - use backward prop to compute dW[l], dbeta[l], dgamma[l] - db[l] is not required as normalization is done
      - Update parameters
        - W[l] - W[l] - alpha * dW[l]
        - beta[l] = beta[l] - alpha * dbeta[l]
        - gamma[l] = gamma[l] - alpha * dgamma[l]
        - can be implemented with momentum, adam, rmspropo
        
![Batch Norm](https://github.com/susantamoh84/DeepLearning/blob/master/Course2/batch%20norm%20as%20regularization.GIF)

# Batch Norm at test time

![Batch Norm Maths](https://github.com/susantamoh84/DeepLearning/blob/master/Course2/batch%20norm%20maths.GIF)

  - At test time the means and std-deviation^2 are calculated for each mini-batch
  - Then using exponentialy weigthed average the overall mean & std-deviation^2
  - Calculate the beta, gamma, z~

# Reference

  - http://cs231n.github.io/neural-networks-2/#reg
