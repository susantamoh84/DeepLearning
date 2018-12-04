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
  
  - Using Word Embeddings
    - You can use words embeddings for Named Entity Recognition
      - Using a large corpus ( 100B words ), you can train the model to understand the relationship between words like apple, oranges
    - Transfer the trained model into a task using smaller word vector
      - Learn word embedding from lage text corpus ( 1-100B words )
      - Download pre-trained embedding online
      - Transfer embedding to new task with smaller training set ( 100K words )
      - Optional: continue to finetune the words embeddings with new data
      
  - Relation to face encoding
    - face recognition: siamese network to compare encodings of 2 different pictures
    - Diff with Words embedding:
      - face recognition takes a new picture and calculates its encoding
      - words embedding learns a fixed embedding ( using the vocabulary )
      
  - By using words embeddings instead of word one-hot vectors, the algorithm can be generalized to learn much better
  
# Properties of word embeddings

  - Analogies:
    - Man --> Woman,  King ---> ? ( Queen ) - Algorithm to learn this
    -         | Man | Woman | King  | Queen | Apple | Orange |
    - Gender  | -1  |  1    | -0.95 | 0.97  | 0.00  | 0.01   |
    - Royal   | 0.01| 0.02  |  0.93 | 0.95  | -0.01 | 0.00   | 
    - Age     | 0.03| 0.02  |  0.70 | 0.69  | 0.03  | -0.02  |
    - Food    | 0.09| 0.01  |  0.02 | 0.01  | 0.95  |  0.97  |
    
    - eman - ewoman ~= [-2 0 0 0]
    - eking - equeen ~= [-2 0 0 0]
      - different man,woman & king,queen is gender
    - Algorithm
      - eman - ewoman ~= eking - e? ( queen )
      
  - Analogies using word vectors:
    - man, woman, kind, queen ---> all are represented as 300-D vectors
    - vector from point man ----> woman & from point kind -----> queen ( represent gender difference )
    - Final word w: arg max ( Sim ( ew, eking-eman+ewoman ) )
    - Similarty function:
      - Sim(u, v) = UtV / ( ||u||2 * ||v||2 )
      - Cosine similarity
    - Examples:
      - Man:Woman as Boy:Girl
      - Ottawa:Canada as Nairobi:Kenya
      - Big:Bigger as Tall:Taller
      - Yen:Japan as Ruble:Russia

# Embedding Matrix

  - In word embedding, the algorithim learns a embedding matrix
  - Total words in vocabulary = 10,000
  - Embedding matrix: 300x10,000
    - O(6267) = [0 0 0 ... 1 (at i=6267) ... 0] <--- One hot representation of the word in the vocabulary
    - E . O(6257) = 300x10K . 10Kx1 = 300x1 
    -             = e(6257)
    - ej = E . Oj
    -    = embedding for word j
    - its not efficient to use a matrix vector multiplication as its high dimensional and most of its elements are 0

# Neural Language Model

  - Using the ej values into a neural network followed by softmax function works very well.
  - Context/target pairs
    - Context:  Last 4 words
    -           4 words on left & right : Example: a glass of orange __ to go along with
    -           Last 1 word             : arrgne __
    -           Nearby 1 word           : glass __  ( Also works well )
    
# Word2Vec Model

  - Skip-grams:
    - I want a glass of orange juice to go along with my cereal
      - Context   : Target
      - orange      Juice
      - orange      glass
      - orange      my
  - Model:
    - Vocab size = 10,000K
    - Context c ("orange") ---> target t ("juice")
    - Oc --> E --> ec --> O ---> yhat ( ec = E.Oc)
    - Softmax: p(t|c) = e^theta*ec/sum(e^theta*ec)
      - Loss: - sum( yi*logyhati )
    - Primary problem:
      - Computationally expensive - softmax computation
    - Hierarchial Softmax
      - computational complexity: order of log(N)
      - common words towards the top
      - less common words deeper in the tree

  - How to sample the context c ?
    - Uniformly random - not efficient as more common words dominate the sample

# Negative Sampling

  - Defining a new learning problem
    - Context     Word      target
    - orange      juice       1
    - orange      king        0
    - orange      book        0
    - orange      the         0
    - orange      of          0
    - For each context pick a target words and label as 1
      - Also pick k random target words and label as 0
      - K = 5-20 for smaller dataset
      - K = 2-5 for larger dataset
    - Here instead of training 10,000 K softmax labels, the model only uses K (2-5) softmax labels
    - P(wi) = ( f(wi)^0.75 / sum( f(wi)^0.75 ) , 1/|V| )
    
# Glove Algorithms

  - Xij = #times i appears in context of j <---- conunt
    - i is the subscript for t
    - j is the subscript for c
  - Model
    - Minimize SUM(i)SUM(j) f(Xij) ( theta(i)t . ej + bi + bj - log Xij ) ^ 2
    - f(Xij) = weightage terms
    - f(Xij) = 0 if Xij = 0
    - thetai , ej are symmetric
    - ew final = (ew + thetaw)/2
    
# Sentiment Classification

  - challenge is not having huge labeled dataset
    - word embedding can help here
  - x - input text
  - y - sentiment rating
  
  - Simple sentiment classification model
    - for each word find the one-hot vector (o) from the dictionary
    - compute ej = E . oj <---- total 300-D, 300 features
    - Average ej 300-D vector
      - By using average operation the algorithm works for short/long review sentences. 
      - Even if the review is 100 words long, sum/average all the feature vector for all 100 words --> 300-D feature representation
    - Feed this to a softmax function 1-5 (possible outcomes) ---> yhat
    
    - RNN for Sentiment classification
      - Compute embedding for all the words in the sentence
      - All of these embeddings are used as input into a RNN model
      - Final activation is fed into a softmax function
      - Many-to-one RNN architecture.
      
    - RNN vs Avergae
      - RNN can identify "not good", "lacking ambiance" as negative sentiment compared to the average model which can't detect it

# Debiasing word embeddings

  - Problem of bias in word embeddings
    - Man:Woman as King:Queen
    - Man:Computer_Programmer as Woman:Homemaker ----> wrong it should be Computer_Programmer
    - Word embedding can reflect gender, ethinicity, age, sexual orientation and other biases of the text used to train the model.
    - It has wide range of implications in real-life scenarios
    
  - 1. Identify bias direction
    - ehe - eshe
    - emale - efemale
    - ...
    - average <---- take average of all the biases 

  - 2. Neutralize: For every word that is not definational, project to get rid of bias
    - doctors, babysitters need to be gender neutral
    - beard need not be neutralized ( Need to select words carefully )
    
  - 3. Equalize Pairs: grandmother, grandfather have exactly same distance ( similarity ) from the words like babysitter
    - pairs are relatively small
    - possible to hand-pick these pairs
    
# Basic Model Examples
  
  ![Translation](https://github.com/susantamoh84/DeepLearning/blob/master/Course5/translation.GIF)
  ![imagecaptioning](https://github.com/susantamoh84/DeepLearning/blob/master/Course5/imagecaptioning.GIF)
  

# Picking the most likely sentence

  ![conditionalmodel](https://github.com/susantamoh84/DeepLearning/blob/master/Course5/conditionalmodel.GIF)
  
  - Why not greedy approach ( pick the best first words, best second word, .... )
    - Because there are more common words which may not be best representation
    
  ![mostlikelytranslation](https://github.com/susantamoh84/DeepLearning/blob/master/Course5/mostlikelytranslation.GIF)    
  
  - Suggested algorithim: arg max P( yi, y2..... yn | x) <----- maximizes the overall probability -- Beam Search
  

# Beam Search Algorithm

  - It's heurestic search ( not exact search )
  - Take the input vector and find the probabilities for the words for the first output yhat1
  - Select the top k words for yhat1 ( beam width = k )
  - For one word from top k words in yhat1, find the most likely words for yhat2 ( yhat1 is fed to the next layer )
    - Select the pair of words (yhat1, yhat2) which gives the maximum probability
      - P( yhat1, yhat2 | x ) = P( yhat1 | x ) * P( yhat2 | x , yhat1 )
    - Do this for all possible combinations of words
    - From all of these combinations, pick the top k which maximizes the probability

# Refinements to Beam Search

  - Length Normalization
    - instead of taking Product ( P( yi, y2..... yn | x) ) take SUM ( log P( yi, y2..... yn | x) )
      - log function makes the algorithim more stable and safe from numberical round errors
    - Normalization: (1/Ty^alpha) *SUM ( log P( yi, y2..... yn | x) )
      - alpha = 1: complete normalization
      
  - How to choose B ?
    - Larger B -- More computationally expensive algorithm, better result
    - Smaller B -- Fast, worse result
    - Try different B values -- 10, 100, 1000, 3000 
    
  - Beam search runs fater BFS, DFS but not guranteed to find the maximum probability

# Error Analysis In Beam Search

  - Errors in different componenets
    - RNN
    - Beam Search
    
  - Y* is human translation
  - Yhat is machine translation
  - In the RNN try to find P ( Y* | x ) & P ( Yhat | x ) and see which one is better
    - By doing above the error can be assigned to either RNN or Beam Search    
    - Case1: P ( Y* | x ) >  P ( Yhat | x )
      - This is Beam Search error as its unable to find the maximum P
      - Increase Beam width
    - Case1: P ( Y* | x ) <=  P ( Yhat | x )
      - This is RNN error as its unable to find the better translation Y*
      - Deeper analysis: reguarization, more training data, network architecture

# Bleau Score

  - Bleau - Bilingual Evaluation Understudy
  - Bleau Score n-gram Pn = sum(n-gram) CountClip ( n-gram ) / sum(n-gram) Count ( n-gram )
  - Combied Bleau Score: BP*exp( 1/4 * sum(n) Pn )
  - BP - Brevity Pentaly
  - BP  = 1 if MT_output_length > reference_output_length
  -     = exp(1 - MT_output_length / reference_output_length ) otherwise
  
# Attention Model Intution

  ![attentionmodel](https://github.com/susantamoh84/DeepLearning/blob/master/Course5/attentionmodel.GIF)    


# Speech Recognition

  - pre-processing: filt_vec output, spectrogram <--- normalizes the audio volumes at different points of time
  - Attention model
  - CTC cost of speech recognition
    - CTC: connectionist temporal classification
  ![speechrecogn](https://github.com/susantamoh84/DeepLearning/blob/master/Course5/speechrecogn.GIF) 
  
  
# Trigger Word System

  - Examples:
    - Amazon Echo: Alexa
    - Google Home: Okay Google
    - Apple Siri: Hey Siri
  - RNN architecure:
    - Immediately after trigger word set the output to 1.
    - Else 0
    - Because of huge imbalace, its good idea to set a few outputs to 1

  ![triggerwords](https://github.com/susantamoh84/DeepLearning/blob/master/Course5/triggerwords.GIF) 

  
  
