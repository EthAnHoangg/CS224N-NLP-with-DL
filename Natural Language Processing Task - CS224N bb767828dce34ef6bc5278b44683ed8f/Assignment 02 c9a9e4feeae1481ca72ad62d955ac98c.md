# Assignment 02

![Untitled](Assignment%2002%20c9a9e4feeae1481ca72ad62d955ac98c/Untitled.png)

# Mathematical proves

# Code Assignment

### Sigmoid function

```python
def sigmoid(x):
    ### YOUR CODE HERE (~1 Line)
    s = 1/(1+np.e**(-x))
    ### END YOUR CODE
    return s
```

### Naive softmax loss and its gradient according to v_c and U

```python
def naiveSoftmaxLossAndGradient(
    centerWordVec,
    outsideWordIdx,
    outsideVectors,
    dataset
):
    """ Naive Softmax loss & gradient function for word2vec models

    Arguments:
    centerWordVec -- numpy ndarray, center word's embedding
                    in shape (word vector length, )
                    (v_c in the pdf handout)
    outsideWordIdx -- integer, the index of the outside word
                    (o of u_o in the pdf handout)
    outsideVectors -- outside vectors is
                    in shape (num words in vocab, word vector length) 
                    for all words in vocab (tranpose of U in the pdf handout)
    dataset -- needed for negative sampling, unused here.
    Return:
    loss -- naive softmax loss
    gradCenterVec -- the gradient with respect to the center word vector
                     in shape (word vector length, )
                     (dJ / dv_c in the pdf handout)
    gradOutsideVecs -- the gradient with respect to all the outside word vectors
                    in shape (num words in vocab, word vector length) 
                    (dJ / dU)
    """

    # ===== loss function =====
    words_in_vocab, word_vector_length = outsideVectors.shape
    y_hat = softmax(outsideVectors.dot(centerWordVec)).reshape((words_in_vocab,1)) #shape (num words in vocab,1)
    y_hat_o = y_hat[outsideWordIdx]
    loss = -np.log(y_hat_o)
    # ======= gradient corresponding to outsides vectors
    observed = outsideVectors[outsideWordIdx].T.reshape(-1,1) #shape (word vector length,) 
    expected = outsideVectors.T.dot(y_hat) #shape (word vector lenth,)
    gradCenterVec = -(observed - expected)
    # ======= gradient corresponding to outsides vectors
    y_hat[outsideWordIdx] -= 1

    # gradCenterVec = outsideVectors.T.dot(y_hat)
    gradOutsideVecs = (y_hat).dot(centerWordVec.reshape((1, word_vector_length))) #(num words in vocab, word vector length) 
    return loss, gradCenterVec.reshape((-1,)), gradOutsideVecs
```

### Negative sampling loss

```python
def getNegativeSamples(outsideWordIdx, dataset, K):
    """ Samples K indexes which are not the outsideWordIdx """
    negSampleWordIndices = [None] * K
    for k in range(K):
        newidx = dataset.sampleTokenIdx()
        while newidx == outsideWordIdx:
            newidx = dataset.sampleTokenIdx()
        negSampleWordIndices[k] = newidx
    return negSampleWordIndices
```

```python
def negSamplingLossAndGradient(
    centerWordVec,
    outsideWordIdx,
    outsideVectors,
    dataset,
    K=10
):
    negSampleWordVectors = outsideVectors[negSampleWordIndices] #(K, word vector length)
    outsideWordVec = outsideVectors[outsideWordIdx]
    z = sigmoid(outsideWordVec.dot(centerWordVec)) #scalar 
    loss = -np.log(z) - np.sum(np.log(sigmoid(-negSampleWordVectors.dot(centerWordVec))))
    ### Please use your implementation of sigmoid in here.
    gradCenterVec = -(1-z)*(outsideWordVec) + negSampleWordVectors.T.dot(1-sigmoid(-negSampleWordVectors.dot(centerWordVec)))
    ### END YOUR CODE
    gradOutsideVecs = np.zeros(outsideVectors.shape)
    gradOutsideVecs[outsideWordIdx] = (z-1) * centerWordVec
    for i in negSampleWordIndices:
        u_k = outsideVectors[i]
        z_prime = sigmoid(-np.dot(u_k,centerWordVec))
        gradOutsideVecs[i] += centerWordVec*(1-z_prime)
    return loss, gradCenterVec, gradOutsideVecs
```

### Skip-gram model

```python
def skipgram(currentCenterWord, windowSize, outsideWords, word2Ind,
             centerWordVectors, outsideVectors, dataset,
             word2vecLossAndGradient=naiveSoftmaxLossAndGradient):
    """ Skip-gram model in word2vec

    Implement the skip-gram model in this function.

    Arguments:
    currentCenterWord -- a string of the current center word
    windowSize -- integer, context window size
    outsideWords -- list of no more than 2*windowSize strings, the outside words
    word2Ind -- a dictionary that maps words to their indices in
              the word vector list
    centerWordVectors -- center word vectors (as rows) is in shape 
                        (num words in vocab, word vector length) 
                        for all words in vocab (V in pdf handout)
    outsideVectors -- outside vectors is in shape 
                        (num words in vocab, word vector length) 
                        for all words in vocab (transpose of U in the pdf handout)
    word2vecLossAndGradient -- the loss and gradient function for
                               a prediction vector given the outsideWordIdx
                               word vectors, could be one of the two
                               loss functions you implemented above.

    Return:
    loss -- the loss function value for the skip-gram model
            (J in the pdf handout)
    gradCenterVec -- the gradient with respect to the center word vector
                     in shape (num words in vocab, word vector length)
                     (dJ / dv_c in the pdf handout)
    gradOutsideVecs -- the gradient with respect to all the outside word vectors
                    in shape (num words in vocab, word vector length) 
                    (dJ / dU)
    """

    loss = 0.0
    gradCenterVecs = np.zeros(centerWordVectors.shape)
    gradOutsideVectors = np.zeros(outsideVectors.shape)
    ### YOUR CODE HERE (~8 Lines)
    centerWordIdx = word2Ind[currentCenterWord]
    centerWordVec = centerWordVectors[centerWordIdx]
    for outsideWord in outsideWords:
        outsideWordIdx = word2Ind[outsideWord]
        loss_, gradCenterVec_, gradOutsideVecs_ = word2vecLossAndGradient(centerWordVec,
                                                                          outsideWordIdx,
                                                                          outsideVectors,
                                                                          dataset)
        loss += loss_ 
        gradCenterVecs[centerWordIdx] += gradCenterVec_
        gradOutsideVectors += gradOutsideVecs_

    return loss, gradCenterVecs, gradOutsideVectors
```

# Some NLP tasks:

## **Named Entity Recognition (NER)**

> Named entity recognition is the first step towards information extraction that seeks to locate and classify named entities in text into pre-defined categories such as the names of persons, organizations, locations, expressions of times, quantities, monetary values, percentages,etc.
> 

## ****Dependency Parsing****

[https://users.soict.hust.edu.vn/huonglt/UNLP/5.3_Dependency Parsing.pdf](https://users.soict.hust.edu.vn/huonglt/UNLP/5.3_Dependency%20Parsing.pdf)

![Untitled](Assignment%2002%20c9a9e4feeae1481ca72ad62d955ac98c/Untitled%201.png)

![Untitled](Assignment%2002%20c9a9e4feeae1481ca72ad62d955ac98c/Untitled%202.png)

![Untitled](Assignment%2002%20c9a9e4feeae1481ca72ad62d955ac98c/Untitled%203.png)

![Untitled](Assignment%2002%20c9a9e4feeae1481ca72ad62d955ac98c/Untitled%204.png)

![Untitled](Assignment%2002%20c9a9e4feeae1481ca72ad62d955ac98c/Untitled%205.png)

![Untitled](Assignment%2002%20c9a9e4feeae1481ca72ad62d955ac98c/Untitled%206.png)