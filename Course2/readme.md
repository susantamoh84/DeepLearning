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

# Dropout Regularization
  
  - Randomly drop-out a portion of the units in each hidden layer  
  ![Dropout](https://github.com/susantamoh84/DeepLearning/blob/master/Course2/dropout.GIF)
