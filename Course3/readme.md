# Machine Learning Strategy

![Strategy Motivation](https://github.com/susantamoh84/DeepLearning/blob/master/Course3/strategy_motivation1.GIF)

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

  - If the ML algorithm is doing worse than human level performance, following can be done:
    - Get Labeled data from human
    - Gain insight from manual error analysis
    - Better analysis of bias & variance
    - Depending on the purpose of the model, the human level error can change:
    ![human level error](https://github.com/susantamoh84/DeepLearning/blob/master/Course3/human%20level%20error.GIF)
    
  - Error analysis example:
    - human proxy for bayes error - 1%
    - Train error - 5%
    - Dev error - 6%
      - Avoidable bias = Train error - human level error = 4%
      - Variance = Dev error - Train error = 1%
        - hence it is advisible here to reduce the training error.
        
  ![supass human level error](https://github.com/susantamoh84/DeepLearning/blob/master/Course3/suprass%20human%20level1.GIF)
    

# Incorrect Labeled Data

  - If random errors, then no need to correct. Leave it as it is.
  - If consistent errors, then its good to fix the errors.
    - Depending on the % of error because of incorrect labeling, you can decide if it needs to be done.
    
  ![Correct labels](https://github.com/susantamoh84/DeepLearning/blob/master/Course3/correct%20dev-test%20set.GIF)
  
# Training & Testing on different distributions

  - 2,00,000 images from source 1 ( high resolution images - webpages ), 10,000 images from source 2( low resolution images - mobile images)
  - Total 2,10,000 images
  - Option 1:
    - Random shuffle on total images and choose train/dev/test sets
    - 2,05,000 images in train, 2500 images each in dev/test set
    - most of the images from the high resolution category in dev/test set.
  - Option 2:
    - Training set - 2,00,000 images from high res, 5000 from low res
    - dev- 2500 images from low res
    - test- 2500 images from low res    
      - you have changed the target as the low res images
  - Option 2 will give better performance over time.
      
  - Training-dev set: Carve out a small portion from training set to be training-dev set.
    - Same distribution as training set, but not used for training
    - Dev & Test set are from a different distribution
  - example;
    - Training error - 1%
    - Training-dev error - 9% ---> variance problem
    - Dev error - 10%
  - example;
    - Training error - 1%
    - Training-dev error - 1.5% 
    - Dev error - 10%  ----------> Data mismatch problem  - learning algorithm hasn't seen data like Dev set.
  - example;
    - Human error - 0%
    - Training error - 10% ----- > Avoidable High bias problem
    - Training-dev error - 11% 
    - Dev error - 12%  
  - example;
    - Human error - 0%
    - Training error - 10% ----- > Avoidable High bias problem
    - Training-dev error - 11% 
    - Dev error - 20% -----------> Data mismatch problem  

# Bias/Variance on mismatched training and dev/test sets

  - Avoidable bias                    = Training set error - Human level error
  - Variance                          = Training-dev set error - Training set error
  - Data Mismatch                     = Dev error - Training-dev error
  - Degree of overfitting in dev set  = Test set error - Dev set error
  
# Addressing data mismatch

  - carry out manual error analysis to try to understand different between training and dev/test sets
  - make training data more similar or collect more data similar to dev/test set.
  - Artificial data synthesis
    - add noise to data
    - generally its a small portion of whole data
    - works significantly well in speech recognition

# Transfer Learning

  - Example1 : if you have a NN trained on image recognition; use it for radiology diagnosis
    - Option 1 - retrain weights/parameters of last layer - fine-tuning
    - Option 2 - retrain all weights/parameters - pre-training ( already trained on image recogn )
  
  - Example2 : you have a NN trained in speech recognition; use it for wakeword/triggerword detection
    - Option 1 - add many more layers 
    - Option 2 - add output layer with many output nodes
    
  - When transfer learning makes sense ?
    - Task A & B have same input x
    - You have lot more data for Task A than Task B
    - Low level features from A could be helpful for learning B
    
# Multi-task learning

  - One NN doing several tasks.
  
  - Simplified autonomous driving example
    - pedestrians
    - cars
    - stop signs
    - traffic lights
      - Y is a (4,m) matrix
      - unlike softmax regression, one image will have multiple labels
  - Can train a big enough NN to do well on all tasks.
  - in practice Multi-task learning is used a lot compared to multi-task learning
  
# End To End Learning

  - Speech Recognition Example
    - X audio -->(MFCC algo) features (ML) --> Phonemes --> words --> y transcript
    - audio --------------------------------------------------------> transcript ( end to end learning )
    - Needs a large dataset to work well. For small data better to use multiple stages

  - Face Recognition
    - zoom into the face from the original image - deep learning
    - recognize the image from the past images - deep learning
    - End to End is to merge the above steps.
    
  - Machine Transalation
    - English --> Text analysis .... --> French
    - English -------------------------> French ( End to End deep learning works well )
    
  - Estimating Child's age from hand images
    - Images --> bones --> bone length --> age
    - Image -----------------------------> age ( requires a lot of data )
    
  - Pro's of Eng to end learning
    - lets the data speak
    - less hand designing of features
  - Cons
    - Need a large amount of data
    - excludes potentially useful hand-designed components
    
