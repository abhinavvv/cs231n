import numpy as np
from random import shuffle
import sys

def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops).

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  dW = np.zeros(W.shape) # initialize the gradient as zero

  # compute the loss and the gradient
  num_classes = W.shape[1]
  num_train = X.shape[0]
  loss = 0.0
  for i in xrange(num_train):
    scores = X[i].dot(W)
    correct_class_score = scores[y[i]]
    for j in xrange(num_classes):
      if j == y[i]:
        continue
      margin = scores[j] - correct_class_score + 1 # note delta = 1
      if margin > 0:
        loss += margin
        # print X[i,:].T.shape, dW[:,y[i]].shape
        # sys.exit(0)  
        dW[:,y[i]] -= X[i,:].T
        dW[:,j] += X[i,:].T

  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train
  dW /= num_train

  # Add regularization to the loss.
  loss += 0.5 * reg * np.sum(W * W)
  dW += reg * W

  #############################################################################
  # TODO:                                                                     #
  # Compute the gradient of the loss function and store it dW.                #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
  #############################################################################


  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero
  num_classes = W.shape[1]
  
  # print X.shape, W.shape
  # scores = W.T.dot(X.T)
  scores = X.dot(W)
  true_scores = scores[np.arange(0, scores.shape[0]),y]
  # correct_class_scores =  * scores[np.arange(0, scores.shape[0]),y]
  correct_class_scores = np.tile(true_scores,(10,1)).T
  # print correct_class_scores.shape
  # scores[np.arange(0, scores.shape[0]),y] = 0
  deltas = np.ones(scores.shape)
  # print scores.shape, correct_class_scores.shape, deltas.shape 
  # sys.exit(0)
  loss = scores - correct_class_scores + deltas

  loss[np.arange(0, scores.shape[0]),y] = 0.0
  
  loss[loss < 0] = 0.0

  overall_loss = float(np.sum(loss))/X.shape[0]
  overall_loss += 0.5 * reg * np.sum(W * W)



  # dW = np.ones(loss.shape)
  dW = np.ones(loss.shape)
  dW[loss <=0  ] = 0
  dW[np.arange(scores.shape[0]),y] = -1 * np.sum(dW, axis=1)
  dW = X.T.dot(dW)
  # #dW = np.dot(X.T, dW)
  # #dW /= X.shape[0]
  # dLds = np.ones_like(loss)
  # dLds[loss<=0] = 0
  # dLds[np.arange(loss.shape[0]), y] = -dLds.sum(axis=1)
  # dLds /= float(X.shape[0])
  # dW = np.dot(X.T, dLds)
  # dW = np.dot(X.T, dLds)
  # #  
  # dW = X.T.dot(loss - np.reshape(np.sum(loss,axis = 1),(-1,1)) * (tmp == 1))
  dW /= float(X.shape[0]) 

  dW += reg * W
  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################
  # pass
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################


  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the gradient for the structured SVM     #
  # loss, storing the result in dW.                                           #
  #                                                                           #
  # Hint: Instead of computing the gradient from scratch, it may be easier    #
  # to reuse some of the intermediate values that you used to compute the     #
  # loss.                                                                     #
  #############################################################################
  # pass
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return overall_loss, dW
