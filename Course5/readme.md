# Sequence Models

  ![Sequence Models](https://github.com/susantamoh84/DeepLearning/blob/master/Course5/SequenceModel.GIF)
  
# Notations

  - "Harry Porter and Hermoine Granger invented a new spell"
  - x: x[0], x[1] ....................................., x[9] <---- all words
  - y: 1, 1, 0........................................., 0    
  - y: y[0], y[1] ....................................., y[9]  <---- entities
  - Tx = length of the input sequence (x) = 9 words
  - Ty = length of the output sequence (y) = 9
    - Ty may not be equal to Tx
  - X(i)<t> <-- t th of the input sequence in training example i
  - Tx(i) <--- input sequence in training example i
  - Y(i)<t> <-- t th of the output sequence in training example i
  - Ty(i) <--- output sequence in training example i
  
# Representing Words

  ![Word Representation](https://github.com/susantamoh84/DeepLearning/blob/master/Course5/WordRepresentation.GIF)
  
  - Standard network doesn't work well
    - feeding 9 inputs to a NN and have 9 outputs <-- is not a good model
  - Input & output's can be different lenghts in different examples
  - Doesn't share features learnt across different positions of the text

# Recurrent Neural Network

  ![RNN Unidirectional](https://github.com/susantamoh84/DeepLearning/blob/master/Course5/RNNUnidirectional1.GIF)
  
  - Waa: Weight of the activations to the next layer
  - Wax: Weight of the input vectors to the next layer
  - Wya: Weight of the activations to the output layer 
  
  - Weakness of RNN: These are uni-directional. The learning from the previous layers get passed on to the next layer. But the learnings from the later words doesn't flow backwards. Example
    - He said, "Teddy Roosevelt was a great president".
    - He said, "Teddy bears are on sale"
      - in these above 2 sentences, the meaning of Teddy cannot be interpreted by the RNN without knowing the later words
      
  - Forward Propagation
    
