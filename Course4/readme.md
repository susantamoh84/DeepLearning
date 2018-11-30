# Convolutional Neural Networks

  - Computer Vision Problems
  ![Computer vision problems](https://github.com/susantamoh84/DeepLearning/blob/master/Course4/computer%20vision.GIF)
  
  - Huge Data Processing
  ![Huge Data](https://github.com/susantamoh84/DeepLearning/blob/master/Course4/huge%20data.GIF)
  
  - Convolution Operation
  ![Convolution](https://github.com/susantamoh84/DeepLearning/blob/master/Course4/convolution.GIF)
  

# Vertical and Horizontal Edge Detection

  - Vertical Convolution filter: 
    - [ [1 0 -1],
        [1 0 -1],
        [1 0 -1] ]

  - Horizontal Convolution filter: 
    - [ [1 1 1],
        [0 0 0],
        [-1 -1 -1] ]

  - Sobel filter:
    - [ [1 0 -1],
        [2 0 -2],
        [1 0 -1] ]

  - Scharr filter:
    - [ [3 0 -3],
        [10 0 -10],
        [3 0 -3] ]

  - The filter having 9 parameters can be learned using back-propagation
    - This approach gives much more robust results compared to hand-picked values

# Padding

  - image (6x6) *(convl) filter(3x3) = output (4x4)
  - image (nxn) *(convl) filter(fxf) = output (n-f+1 x n-f+1) <-- dimension of image is shrinking by convolution
  - 2 problems
    - Shrinking output
    - throwing aways information from edges of image
  - Solution - padding
    - before applying convolution, padd the image with additional 1 pixel on the border
    - p = 1
    - 6x6 image after padding --> 8x8
    - output (n+2p-f+1 x n+2p-f+1)
    - applying filter 3x3 , output --> 6x6 ( original image size is retained )
    - padded cells have value 0
  - Valid Convolutions:
    - no padding: image (nxn) *(convl) filter(fxf) = output (n-f+1 x n-f+1)
  - Same Convolutions:
    - output size = input size
    - n+2p-f+1 = n => p = (f-1)/2
    - f usually odd
    
# Strided Convolutions

  - While calculating element wise product between input & filter, 
    - for each cell in output, the input layer jumps by 2 instead of 1.
    - stride = 2
  - Strided : image (nxn) *(convl) filter(fxf) = output ( floor((n+2p-f+1)/s + 1) x floor((n+2p-f+1)/s + 1) )

# Convolutions Over Volumne

  - RGB Image (6x6x3) *(convl) filter(3x3x3) = output ( 4x4 ) --- output is 2-D
  - multiple filters:
    - input * vertical filter -> output1 (4x4)
    - input * horizontal filter -> output2 (4x4)
    - stack output1, output2 together -> output (4x4x2)
    - image (NxNxNc) *(convl) filter(fxfxNc) = output ( (n-f+1)/4 x (n-f+1)/4 x Nc )
      - Nc- number of channels or depth of volumne
      
# One-layer Convolutional Network

  - Example
    - input 6x6x3 A[0] * 3x3x3 filter1 W[1] --> 4x4 output1 --> relu ( output1 4x4 + b1 ) --> 4x4
    - input 6x6x3 * 3x3x3 filter2 --> 4x4 output2 --> relu ( output2 4x4 + b2 ) --> 4x4
    - stack output of activations --> 4x4x2 --> A[1]
    - Z[1] = W[1] (filters) * A[0] + b1
    - A[1] = g(Z[1]) -- g is relu
    - if 10 filters, output = 4x4x10
    - Number of parameters:
      - 10 filters are siz 3x3x3 + bias
      - 10*28 parametrs
      - 280 parameters

  - Example ConvNet
    - Image: 39x39x3 
      - Nh = Nw = 39, Nc = 3
    - 10 filters
      - f[1] = 3, S[1] = 1, P[1] = 0
    - Output: 37x37x10 ( (n+2p-f)/S + 1 )
      - Nh[1] = Nw[1] = 37, Nc = 10
    - 20 filters
      - f[1] = 5, S[1] = 2, P[1] = 0
    - Output: 17x17x20
      - Nh[2] = Nw[2] = 17, Nc = 20
    - 40 filters
      - f[1] = 5, S[1] = 2, P[1] = 0
    - Output: 7x7x40
      - Nh[2] = Nw[2] = 17, Nc = 10
    - flatten last output and apply logistic/softmax:Yhat

  - Types of layer in Conv Network
    - Convolution
    - Pooling 
    - Fully Connected
    
# Pooling

  - Pooling layer: Max pooling
    - input 4x4 --> output 2x2 , f = 2, s = 2   
    - [[ 1 3 2 1 ],   ----->    [[ 9 2 ]
       [ 2 9 1 1 ],              [ 6 3 ]]
       [ 1 3 2 3 ],
       [ 5 6 1 2 ]]
    - Applies Max function to the quadrants
    - hyper params - f,s
    - No parameters to learn
    - input 5x5xNc --> output 3x3xNc, max pooling done on each of the Nc channels
    
  - Average Pooling: Take Average of numbers
    - similar to max pooling
    - input 7x7x1000 ----> 1x1x1000
    
# CNN Example: ( LeNet-5)

  - Input 32x32x3
  - Layer 1
    - f=5,s=1 Conv1 - 28x28x6
    - maxpool f=2,s=2 Pool1 - 14x14x6
  - Layer 2
    - f=5,s=1 Conv2 - 10x10x16
    - maxpool f=2,s=2 Pool2 - 5x5x16
  - Flatten 400 outputs into single layer
  - First Fully Connected Layer 120 units ( W[3] - 120x400, b[3] - 120x1
  - Fully Connected Layer 84 units
  - Softmax 10 outputs
  
  - Architecture
    - CONV-POOL-CONV-POOL-FC-FC-FC-SOFTMAX
    
# Why Convolutions

  - if image 32x32x3 fully connecting to 28x28x6
    - Number of parameters to train = 3072x4704 = 14million - huge number of parameters
  - In convolution layer filter=5x5 + 1 bias = 26 parameters
    - 6 filters = total parameters = 6*26 = 156 parameters --------> very small parameters compared to FC
  - Parameter Sharing
    - feature detector ( like vertical edge detector ) which is useful in one part of image is probably useful in another part of the image
    - Sparsity of connections: Each output only depends on a smaller number of inputs
    - translation invariance - image shifted by a few pixels
    

# Networks Case
  - Classic Networks
    - LetNet-5
    - AlexNet
    - VGG
  - ResNet
  - INception
  
# LetNet-5

  ![LetNet-5](https://github.com/susantamoh84/DeepLearning/blob/master/Course4/LeNet.GIF)
  - Nh, Nw decreases along the network and Nc increases
  - 60K parameters to tune
  
# AlexNet

  ![AlexNet](https://github.com/susantamoh84/DeepLearning/blob/master/Course4/AlexNet.GIF)
  - Large Network
  - 60million parameters to tune
  
# VGG-16

  ![VGG](https://github.com/susantamoh84/DeepLearning/blob/master/Course4/VGG.GIF)
  - Very large but uniform network
  - 138 million parameter
  - 16 layers of conv+pool layers
  
# ResNet

  - Residual Block:
    - a[l] --> linear --> Relu a[l+1] --> linear - * -> Relu a[l+2]
        |                                          |  
        -------------------------------------------| <--- shortcut
    - Short-cut: a[l+2] = g( z[l+2] + a[l] )
    - Allows to train much much deeper networks
      - because of short-cuts, the gradients don't explode/vanish so fast
    ![ResNet](https://github.com/susantamoh84/DeepLearning/blob/master/Course4/ResNet.GIF)
    
# Using 1x1 convolutions

  - Reduces the number of channels
  - Adds more non-linearity to the network
  - Example:
    - Input 28x28x192 
    - 32 1x1 conv : 1x1x192
    - Output: 28x28x32 <---- Reduced the number of channels from 192 to 32

# Inception Network

  ![Inception Net](https://github.com/susantamoh84/DeepLearning/blob/master/Course4/Inception.GIF)
  
  - Computation Cost is high
  - Estimation of compute cost
    - Example image 28x28x192 ---- Conv (5x5) 32 filters, same ---> Output 28x28x32
    - cost - 28x28x32 * 5x5x192 --- 120 million parameters
  - Reduce cost using 1x1 conv
    - 28x28x192 ---- Conv (1x1) 16 filters, same ---> image 28x28x16 ---- Conv (5x5) 32 filters, same ---> 28x28x32
    - compute cost: 28x28x16 * 1x1x192 = 2.4 million; 28x28x32 * 5x5x16 = 10.0 million; total cost = 12.4 million
    - 1/10 th of the previous cost
    - Conv 1x1 is called Bottleneck layer
  - Bottle neck layer does't impact learning provided used carefully
  - Inception network - Inception module getting repeated multiple times
  - GoogleNet - multiple Inception networks stacked
  
# Data Augumentation Techniques

  - Mirroring ---> common
  - Random Cropping ---> common
  - Roation
  - Shearing
  - Local Wrapping
  
  - Color Shifting
    ![Color Shifting](https://github.com/susantamoh84/DeepLearning/blob/master/Course4/ColorShifting.GIF)

# State Of Computer Vision

  - Because of lack of data, hand engineering is very important here.
  - Transfer Learning can be useful
  ![State ComputerVision](https://github.com/susantamoh84/DeepLearning/blob/master/Course4/StateOfComputerVision.GIF)
  
  - Tips For Winning Competetions
    - Ensemble - many model, average output
    - Multi-crop test data - Tests on multiple version of the test data and aggregate
    
# Object Detection/Localization

  ![Object Detection](https://github.com/susantamoh84/DeepLearning/blob/master/Course4/ObjectDetection.GIF)
  
  - Classification (Normal)
    - Image ----> Convl layer ----> Softmax 4 units ( for classifying pedestrian, car, motorcyle, background )
  - To find the location of car inside the image
    - bounding box parameters: 
      - bx, by - centroid of the bounding box
      - bh, bw - height, width of the bounding box
    - Add these parameters to the output layer
      - output layer= [probablity] classification objects + bounding box parameters
      - y = [ Pc, bx, by, bh, bw, c1, c2, c3 ]
        - Pc=1 if there is an object else 0
  - Loss function L(yhat, y) = sum[ (yhat1-y1)^2 + (yhat2-y2)^2 + (yhat3-y3)^2 ....yhat8-y8)^2 ] if y1=1
                             = (yhat1-y1)^2 if y1=0

# LandMark Detection

  ![LandMark Detection](https://github.com/susantamoh84/DeepLearning/blob/master/Course4/LandMarks.GIF)
  
# Sliding Windows Detection

  - pass a number of windows of various sizes and see if an object is detected
  - high computational cost

# Convolutional Layers ( used for sliding windows )

  - Convolutional implementation
  ![Convl Layer](https://github.com/susantamoh84/DeepLearning/blob/master/Course4/SlidingWindowsConvLayer.GIF)
  - Sliding Window working
  ![Convl Layer](https://github.com/susantamoh84/DeepLearning/blob/master/Course4/SlidingWindowsConvLayerWorking.GIF)
  - Bounding box is less accurate
  
# Yolo Algorithim ( more accurate bounding boxes )
      
  - Yolo - You only look once
  - Place a grid on the image - 3x3, 19x19 etc
  - Output layer - 3x3x8 volume ( for a grid of 3x3 )
  - Very fast - as it need only 1 conv net
  - used in real-time object detection
  ![Convl Layer](https://github.com/susantamoh84/DeepLearning/blob/master/Course4/Yolo.GIF)
  - bounding boxes should be between 0,1
  - But bounding boxes can be > 1
  
# Intersection Over Union

  - Computes Intersection Over Union of the bounding boxes [ Used for the running the convul network ]
  - IOU ( Intersection Over Union ) = (Size of Intersection of the boxes) / (Size of Union of the boxes)
    - Correct if IoU >= 0.5 - the answer is decent
      - this is just a convention
      - higher the IoU, the better the accuracy of the algorithim

# Non-max supression example

  - When the grid cells are large, many cell will detect the car
  - non-max supression removes the multiple detection of the cars and only marks 1 car detection
    - Discard all boxes with Pc <= 0.6
    - Finds the probability of the car detection (Pc) which is the heighest
      - Pickup the box with the largest Pc, output that as a prediction
      - Discard any remaining box with IoU >= 0.5 with the box output in the previous step

# Anchor Boxes

  - Overlapping Objects: When there are more than 1 predictions, anchor boxes are required
  - 2 anchor boxes ( horizontal rectange or vertical rectangle )
    - Y = [ Pc, bx, by, bh, bw, c1, c2, c3, Pc,bx...... C3 ]
  - Anchor Box Algorithim:
    - Previously: each object in training image is assigned to grid cell containing object's midpoint.
      - Output y: 3x3x8
    - Anchor Boxes: 
      - each object in training image is assigned to grid cell containing object's midpoint.
      - Also assigned to the anchor box for the grid cell with the highest IoU
        - output y: 3x3x16 (2x8) --- > 2 anchor boxes
          - dimension of y = 8 because of 3 different classes
      - Car Only (horizontal anchor box) Y = [ Pc, bx, by, bh, bw, c1, c2, c3, Pc,bx...... C3 ]
                                             [ 0, ??                          , 1 , bx ........]
      -                                        ----------anchor box1---------   ----- anchor box2--
      - doesn't work well for more than 2 objects overlapping
      
# YOLO Algorithm

  - y = 3x3x2x8 - for 3 classes prediction [ 1 - pedestrian, 2 - car, 3 - motorcycle ]
    - 3x3 is for a grid of size 3x3 ( In-practice it'll be 19x19 or even larger )
    - 2 is for anchor boxes
  - Outputting the non-max supressed outputs
    - For each cell in grid (3x3), there will be 2 bounding boxes predicted
    - it can go outside the grid cell height & widht
    - Get rid of low probability predictions
    - For each class ( pedestrian, car, motorcycle ) use non-max supression to generate final predictions.
    
# Region proposal: R-CNN

  - Pick few regions to run the Conv-net classifier
  - Region proposal:
    - Segmentation algorithm: Find ~ 2000 blobs and run classifier on those
  - Region proposal & classify proposed regions one at a time.
  - It is slow.
    - Fast R-CNN - Use convolutions impl of sliding windows to classify all proposed regions
      - Region proposal is still slow
    - Faster R-CNN - Use Conv net to propose regions

# Face Recognition
  
  - Liveness Detection - This is one important aspect of identifying the whether the face recognition can work on a photograph or live person
  
  - Face Recognition Problem
    ![Face Recogn](https://github.com/susantamoh84/DeepLearning/blob/master/Course4/faceRecogn.GIF)
    
# One-Shot Learning

  - Learning from just 1 image of the person in the database
    - Use CNN with softmax.
    - Doesn't work well because of less training data
  - Learning "Similiarity" function
    - degreee of difference  = d(img1, img2)
    - if degree of difference <= T - same person        }   <------ Verification
    -                         >  T - different person   } 

# Siamese Network

  - Neural Network having same weights is used to create encoding vectors for different images
  ![Deep Face](https://github.com/susantamoh84/DeepLearning/blob/master/Course4/Siamese.GIF)
  
  - Learning Objective of Siamese Network
  ![Siamese Learning](https://github.com/susantamoh84/DeepLearning/blob/master/Course4/SiameseLearning.GIF)
  
# Tripplet Loss

  - A = Anchor image
  - P = Positive image match
  - N = Negative image match
  - || f(A) - f(P) ||^2  + alpha <=  || f(A) - f(N) ||^2
  - Example 
    - d(A,p) = 0.5 
    - d(A,N) = 0.51
    - not good enough to say a negative match. d(A,N) should be greater than 0.7 to confirm the negative match.
  - Loss L(A,P,N) = MAX( || f(A) - f(P) ||^2  -  || f(A) - f(N) ||^2 + alpha, 0 ) , always >= 0
    - J = sum[i] L(A[i], P[i], N[i] )
  - Example training set: 10k pictures for 1k persons    
  - Choosing A,P,N for the training set
    - If choosen randomly, then d(A,P) + alpha <= d(A,N) is easily satisfied
    - Choose the triplets when d(A,P) + alpha <= d(A,N) is hard to satisy
      - increases efficieny of the gradient descent algorithm
  - Algorithims
    - Face Net
    - Deep Face
  ![Triplet Training](https://github.com/susantamoh84/DeepLearning/blob/master/Course4/TripletTraining.GIF)
  
  - In industry companies are using 10,100million images for face nets
  
# Face Verification & Binary Classification

  - Input pair of image
  - Applying simaese learning, find the 128 encoded vector for each image
  - Take the different of both the encoded vectors for 2 different imag
  - Run a logistic regression on the different vectors to find the similarity
  - Pairs Of Images ====> target labels (0/1) <------ Supervised learning
    - Train a siamese network using back propagation
