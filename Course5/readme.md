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
    - a[0] = 0 vector
    - a[1] = g1 ( Waa * a[0] + Wax * x[1] + ba )  <------ activation is tanh/Relu function
    - y[1] = g2 ( Wya * a[1] + by )               <------ activation is sigmoid
      - Notation: a <- Wax * x <---- Wax meaning weight to be multiplied to x to get output of a
    
    - In-general:
    - a< t > = g1 ( Waa * a< t-1 > + Wax * x<t> + ba )
    - y< t > = g2 ( Wya * a< t > + by ) 
    
    - Simplified further:
    - a< t > = g1 ( Wa [ a< t-1 > , x< t > ] + ba )
      - Wa = [ Waa | Wax ]
      - Waa - 100x100
      - Wax - 100x10000
      - Wa  - 100x10100
      - [ a< t-1 > , x< t > ] - stacking the vectors together [ a< t-1 > ,
      -                                                   x< t >  ]  <--- 10100
    - y< t > = g2 ( Wy [ a< t > ] + by ) 
    
    - Loss Function:
      - L< t > = -y< t > * logy< t > - (1 -y< t >) * log(1 -y< t >)
        - for a single sequence ( single word in word represntation )
      - L = sum( L< t > )
      
# Different types of RNN

  - Word Representation: Many to Many ( Many input sequence to many output sequence )
  - Sentiment Classification: Many to One ( Many input sequence to one output )
  - Music Generation: One to Many ( input is just one sequence - genre etc )
  - Machine Translation: Many to Many ( input & output sequences are of different length )
    - encoder & decoder layers
 
 ![RNN Types](https://github.com/susantamoh84/DeepLearning/blob/master/Course5/RNNTypes.GIF)
    

# Language Model & Sequence Generation

  ![Language Model](https://github.com/susantamoh84/DeepLearning/blob/master/Course5/LanguageModel.GIF)
  
  - Language Modelling in RNN
    - Training set: Large corpus of english text
      - corpus: large body/amount of text
    - Tokenize Words
      - Can add < EOS > - End of sentence token at the end of the sentence
      - Add punctuations (.) to the vocabulary
    - Create one-hot encoding of the entire vocabulary ( of all words )
    - Unknown words are mapped as < UNK > 
    - In the language RNN model X< t > = y< t-1 >
    
    ![Lanugage RNN](https://github.com/susantamoh84/DeepLearning/blob/master/Course5/LanguageModelRNN.GIF)
    ![Charachter RNN](https://github.com/susantamoh84/DeepLearning/blob/master/Course5/CharachterRNN.GIF)
    ![LanguageRNNEx RNN](https://github.com/susantamoh84/DeepLearning/blob/master/Course5/LanguageRNNEx.GIF)
    
# RNN Vanishing Gradient

  - RNN are strongly influenced by local inputs than the input far earlier in the sequence

# Gated Recurrent Unit

  - Solves the vanishing gradient problem
  - captures the long range influences in the long running sequences
  
  ![GRU](https://github.com/susantamoh84/DeepLearning/blob/master/Course5/GRU.GIF)
  ![Full GRU](https://github.com/susantamoh84/DeepLearning/blob/master/Course5/FullGRU.GIF)
  
  - when Tau ---> 0
    - Ct ~ Ct-1
    - The previous weights have a strong influence
  
# LSTM ( Long Short Term Memory )

  ![GRU LSTM](https://github.com/susantamoh84/DeepLearning/blob/master/Course5/GRUvsLSTM.GIF)
  ![LSTM](https://github.com/susantamoh84/DeepLearning/blob/master/Course5/LSTM.GIF)
  
  - PeeHole Connection
    - Where the gate connection have a, x & c< t-1 > vectors
    
  - GRU vs LSTM
    - GRU is a more recent invention
    - researchers try both
    - GRU is much simpler model than LSTM
    - LSTM has 3 gates compared to GRU 2 gates and is more powerful and flexible
    
# Bi-directional RNN

  ![BRNN](https://github.com/susantamoh84/DeepLearning/blob/master/Course5/BRNN.GIF)
  
  - The advantage - both previous & future words
  - Dis-advantage - need full sequence of words before predictions; computionally expensive
    - Doesn't work well in real-time words prediction
    - But some applications where full sentence can be extracted, works well.
        
# Deep RNN

  - Maximum 3 deep re-current layers ( because of computational expensiveness )
  - Beyond this there can be more deep layers for y values
  - Deep RNN block can be also LSTM, GRU, BRNN
  
  ![Deep RNN](https://github.com/susantamoh84/DeepLearning/blob/master/Course5/DeepRNN.GIF)

# Word Embeddings

  ![Word Features](https://github.com/susantamoh84/DeepLearning/blob/master/Course5/WordFeature.GIF)
  ![Word Embeddings](https://github.com/susantamoh84/DeepLearning/blob/master/Course5/WordEmbedding.GIF)
  
  - Visualization algorithim is t-sne
  - it maps the point from 300-D space to 2-D space
  
