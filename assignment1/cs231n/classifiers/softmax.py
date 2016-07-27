import numpy as np
from random import shuffle
import sys

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

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
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  num_train = X.shape[0]
  num_class = W.shape[1]
  p = np.zeros((num_class,1))
  for i in range(num_train):
    score = X[i,:].dot(W)
    correct_class_score = score[y[i]]
    # score[y[i]] = 0
    const = -np.max(score)
    score += const
    p = np.exp(score)/np.sum(np.exp(score))
    loss += -np.log(p[y[i]])
    p[y[i]] -= 1 

    dW += np.dot(np.reshape(X[i],(W.shape[0],1)), np.reshape(p,(1,num_class)))


  loss /= num_train
  loss += 0.5 * reg * np.sum(W * W)
  dW = (1.0/num_train)*dW + reg*W


  # for i in X.shape[0]:
  #   f_x = X[i].dot(W)
  #   correct_class_score = f_x[y[i]]
  #   f_x[y[i]] = 0.0
  #   const = -np.max(f_x)
  #   f_x += const
  #   correct_class_score += const 
    
  #   p = np.exp(correct_class_score)/np.sum(np.exp(f_x))
  #   loss += -np.log(p)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  
  num_train = X.shape[0]
  scores = X.dot(W)
  # print np.max(scores, axis=1).shape
  # sys.exit(0)
  scores -= np.tile(np.max(scores, axis=1),(scores.shape[1],1)).T
  # print scores.shape
  normalized_scores = np.exp(scores)/np.tile(np.sum(np.exp(scores),axis=1),(scores.shape[1],1)).T
 
  loss = np.sum(-np.log(normalized_scores[range(num_train),y]))
  loss = loss / num_train + 0.5 * reg * np.sum(W * W)


  gradient_score = normalized_scores
  gradient_score[range(num_train),y] -= 1
  dW = X.T.dot(gradient_score) / num_train + reg * W 

  # sys.exit(0)


  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

