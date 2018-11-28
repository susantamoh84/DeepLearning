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
    
