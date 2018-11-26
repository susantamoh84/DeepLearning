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

    
