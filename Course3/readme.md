# Machine Learning Strategy

![Strategy Motivation](https://github.com/susantamoh84/DeepLearning/blob/master/Course3/strategy_motivation.GIF)

# Orthogonalization

  - Each control only affects one part of the outcome
  
  - Chain of Assumptions in ML
    - Fit training set well on cost function ( ~ Humal Level performance ) - bigger network, adam optimization, 
      - don't use early stopping initially
    - Fit dev set well on cost function - Regularization, Bigger Training Set
    - Fit test set well on cost function - Bigger dev set
    - Performs well in real-world - Change dev set, cost function
    
# Single Number Evaluation Metric

  - Precision - Of examples recognized as cats, what % are actually cats ?
  - Recall - What  of the actual cats are recognized ?
  - F1 score - Average of precision and recall - 2/(1/P+1/R) - Harmonic mean
  
  - N metrics - 1 optimizing, N-1 satisfying
  
# Splitting Data

  - For Data size - 100, 1000, 10,000
    - 70% train, 30% test
    - 60% train, 20% dev, 20% test
  - For Data size 1,000,000
    - 98% train, 1% dev, 1% test
    
# Compare with human level performance

  - Its possible to get human level accuracy
  - Minimum possibe error is Bayes Optimal Error - best possible error
![Error Compare](https://github.com/susantamoh84/DeepLearning/blob/master/Course3/compare%20human%20level.GIF)
