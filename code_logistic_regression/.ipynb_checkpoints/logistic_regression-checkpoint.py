########## >>>>>>Nick Adams 00883496

# Implementation of the logistic regression with L2 regularization and supports stachastic gradient descent
import numpy as np
import math
import sys
sys.path.append("..")
from code_misc.utils import MyUtils

class LogisticRegression:
    def __init__(self):
        self.w = None
        self.degree = 1

        

    def fit(self, X, y, lam = 0, eta = 0.01, iterations = 1000, SGD = False, mini_batch_size = 1, degree = 1):
        ''' Save the passed-in degree of the Z space in `self.degree`. 
            Compute the fitting weight vector and save it `in self.w`. 
         
            Parameters: 
                X: n x d matrix of samples; every sample has d features, excluding the bias feature. 
                y: n x 1 vector of lables. Every label is +1 or -1. 
                lam: the L2 parameter for regularization
                eta: the learning rate used in gradient descent
                iterations: the number of iterations used in GD/SGD. Each iteration is one epoch if batch GD is used. 
                SGD: True - use SGD; False: use batch GD
                mini_batch_size: the size of each mini batch size, if SGD is True.  
                degree: the degree of the Z space
        '''
        # setting global degree 
        self.degree = degree
        # z_transform samples
        samples = MyUtils.z_transform(X, self.degree)
        # add bias feature to samples 
        samples = np.insert(samples, 0, 1, axis=1)
        # concatenating samples with their labels before shuffling
        #samples_labels = np.concatenate((samples.T, y.T)).T # add this back in to shuffle data
        # shuffle samples randomly in place
        #np.random.shuffle(samples_labels) # add this back in to shuffle data
        
        # initialize w vector with zeros
        n, d = samples.shape
        self.w = np.zeros((d, 1))
        
        shuffled_y = y #samples_labels.T[d].reshape(n, 1) # add this back in when shuffling data
        shuffled_X = samples #np.delete(samples_labels, d, axis=1) # add this back in when shuffling data 
        
        X_prime = shuffled_X
        y_prime = shuffled_y
        n_prime = n
        current_step = 0
        while(iterations > 0):
            if(SGD == True):
                current_step += mini_batch_size
                # currently not shuffled data look above if you wish to shuffle
                current_step, X_prime, y_prime, n_prime = self.getMiniBatch(shuffled_X, shuffled_y, current_step, mini_batch_size)
            
            s = (y_prime * (X_prime @ self.w))
            self.w = (eta/n_prime) * ((y_prime * LogisticRegression._v_sigmoid(-s)).T @ X_prime).T + \
                     (1 - 2 * eta * lam / n_prime) * self.w 
            iterations -= 1
   
    def getMiniBatch(self, X, y, current_step, mini_batch_size):
        n,d = X.shape
        n_prime = mini_batch_size
        min_index = current_step - mini_batch_size
        
        if(n - current_step < 0):
            # If we step over all the samples keep up to n-1 then wrap to top 
            #  and concatenate remaining mini_batch to bottom.
            n_max_index = mini_batch_size - (n - min_index)
            X_prime = np.concatenate((X[min_index : n], X[0 : n_max_index]))
            y_prime = np.concatenate((y[min_index : n], y[0 : n_max_index]))
            current_step = 0
        else:
            # Else we just need from min_index to current_step which will be mini_batch samples in size
            X_prime = X[min_index : current_step]
            y_prime = y[min_index : current_step]
        
        return current_step, X_prime, y_prime, n_prime
    
    def predict(self, X):
        ''' parameters:
                X: n x d matrix; n samples; each has d features, excluding the bias feature. 
            return: 
                n x 1 matrix: each row is the probability of each sample being positive. 
        '''
    
        # remove the pass statement and fill in the code. 
        samples = MyUtils.z_transform(X, self.degree)
        samples = np.insert(samples, 0, 1, axis=1)
        
        return self._v_sigmoid(samples @ self.w)
    
    def error(self, X, y):
        ''' parameters:
                X: n x d matrix; n samples; each has d features, excluding the bias feature. 
                y: n x 1 matrix; each row is a labels of +1 or -1.
            return:
                The number of misclassified samples. 
                Every sample whose sigmoid value > 0.5 is given a +1 label; otherwise, a -1 label.
        '''

        # remove the pass statement and fill in the code.
        prediction_set = self.predict(X)
        
        error = 0
        for i in range(len(prediction_set)):
            if(prediction_set[i] > 0.5 and y[i] == -1) or (prediction_set[i] <= 0.5 and y[i] == 1):
                error += 1
                
        return error
    
    @staticmethod
    def _v_sigmoid(s):        
        v_sigmoid = np.vectorize(LogisticRegression._sigmoid)
        return v_sigmoid(s)
    
    @staticmethod
    def _sigmoid(s):        
        sigmoid = 1 / (1 + np.exp(-s))
        return sigmoid

