import numpy as np
from datetime import datetime


class MyLogisticRegression:
    def __init__(self, learning_rate = 1, num_iterations = 2000, regularization = None, C = None, initialization = None):
        """
        The parameters of logistic regression class.
        
        Argument:
        learning_rate -- the parameter that influense on vector of gradients (by default = 1)
        
        num_iterations -- number of steps in order to get best convergense of function
        
        regularization -- type of regularization "l1" or 'l2' (by default no regularization)
        
        C -- the strength of regularization (by default none)
        
        """
        self.initialization = initialization
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.C = C
        self.regularization = regularization
        if self.regularization:
            if self.C is None:
                self.C = 0.01
            else:
                self.C = C
        self.w = []
        self.b = 0

    def initialize_weight(self, dim):
        """
        This function creates weights vector of shape (1,dim) for w and initializes b = 0.
        Argument:
        dim -- size of the w vector we want (or number of parameters in this case)
        
        """
        if self.initialization == "xavier":
            self.w = np.random.normal(loc=0.0, scale = np.sqrt(2/(1+dim)), 
                                        size = (1,dim))
        elif self.initialization == "normal":
            self.w = np.random.normal(size = (1,dim))
            
        else:
        
            self.w = 0.01 * np.random.randn(1,dim)

        self.b = 0
        return self.w, self.b


    def sigmoid(self, z):
        """
        Compute the sigmoid of z
        Argument:
        z -- is the decision boundary of the classifier
        
        """
        s = 1 / (1 + np.exp(-z))
        return s

    def hypothesis(self, w, X, b):
        """
        This function calculates the hypothesis for the present model
        
        Argument:
        w -- weight vector
        X -- The input vector
        b -- The bias vector
        """
        H = self.sigmoid(np.dot(X, w.T) + b)

        return H

    def cost(self, H, Y, m, w, C):
        """
        This function calculates the cost of hypothesis
        
        Arguments:
        H -- the hypothesis vector
        Y -- the output
        m -- number of training samples
        w -- weights
        C -- strength of regularization
        """
        m = Y.shape[0]
        if self.regularization == 'l1':
            cost = -np.sum(Y * np.log(H) + (1 - Y) * np.log(1 - H)) / m + np.sum(w)*C/(m)
        elif self.regularization == 'l2':
            cost = -np.sum(Y * np.log(H) + (1 - Y) * np.log(1 - H)) / m + np.sum(np.square(w))*C/(2*m)
        else:
            cost = -np.sum(Y * np.log(H) + (1 - Y) * np.log(1 - H)) / m

        cost = np.squeeze(cost)
        return cost

    def cal_gradient(self, w, H, X, Y, C):
        """
        Calculates gradient of the given model in learning space with given regularization parameter

        
        """
        m = X.shape[0]
        if self.regularization == 'l1':
            dw = (np.dot((H - Y).T, X) + (C * np.sign(w)))/m
        elif self.regularization == 'l2':
            dw = (np.dot((H - Y).T, X) + (C * w))/m
        else:
            dw = np.dot((H - Y).T, X)/m
        db = np.sum(H - Y) / m

        return dw, db

    def gradient_position(self, w, b, X, Y, C):
        """
        It just gets calls various functions to get status of learning model
        Arguments:
        w -- weights
        b -- bias, a scalar
        X -- data
        Y -- true "label" vector (containing 0 or 1 )
        C -- strength of regularization
        
        Returns:

        dw —- gradients of weights
        db -- gradients of biases
        cost -- costs
        """

        m = X.shape[1]

        H = self.hypothesis(w, X, b)  # compute activation
        cost = self.cost(H, Y, m, w, C)  # compute cost
        dw, db = self.cal_gradient(w, H, X, Y, C)  # compute gradient
        return dw, db, cost

    def gradient_descent(self, w, b, X, Y, batchsize):
        """
        This function optimizes w and b by running a gradient descent algorithm and store the weigths in self

        Arguments:
        w — weights, a numpy array of size (num_px * num_px * 3, 1)
        b — bias, a scalar
        X -- data of size (no. of features, number of examples)
        Y -- true "label" vector (containing 0 or 1 ) of size (1, number of examples)
        batchsize -- size of batches


        Returns:

        costs — list of all the costs computed during the optimization.
        """
        # Cost and gradient calculation
        costs = []
        results = []
        time_elapsed_list = []

        for i in range(self.num_iterations):
            time_start_epoch = datetime.now()

            batch_cost = []
            for x_batch, y_batch in self.iterate_minibatches(X, Y, batchsize):
                dw, db, cost = self.gradient_position(w, b, x_batch, y_batch, self.C)
                w = w - (self.learning_rate * dw)
                b = b - (self.learning_rate * db)

                self.w = w
                self.b = b

        return costs

    def fit(self, X_train, Y_train, batchsize):
        """
        Function that runs the learning process

        Arguments:
        X_train -- whole matrix of train data
        Y_train -- whole matrix of groun truth
        batchsize -- size of batches


        Returns:

        This function returns nothing but an optimized weights
        """
        
        dim = np.shape(X_train)[1]
        
        w, b = self.initialize_weight(dim)

        cost = self.gradient_descent(w, b, X_train, Y_train, batchsize)



    def get_param(self):
        '''
        Return trained parameters. Call this function to get the trained weigths

        '''
        return {"w": self.w,
                 "b": np.array(self.b).reshape((1, 1))}

    def print_param(self):
        '''
        Function that prints out all the arguments of the logistic regression function

        '''
        print(f'learning rate is: {self.learning_rate}\n'
              f'number of iteration is: {self.num_iterations}\n'
              f'regularization: {self.regularization}\n'
              f'regularization strength: {self.C}')

    def predict(self, X):
        '''
        Predict whether the label is 0 or 1 using learned logistic regression parameters (w, b)

        Arguments:

        X -- data of size (number of examples * num_features)

        Returns:
        Y_prediction -- a numpy array (vector) containing all predictions (0/1) for the examples in X
        '''
        m = X.shape[0]
        X = X.toarray()

        Y_prediction = np.zeros((m, 1))

        w = self.w
        b = self.b
        # Compute vector "H"
        H = self.hypothesis(w, X, b)
        for i in range(H.shape[0]):
            # Convert probabilities H[i,0] to actual predictions p[i,0]
            if H[i, 0] >= 0.5:
                Y_prediction[i, 0] = 1
            else:
                Y_prediction[i, 0] = 0

        return Y_prediction.T

    def predict_proba(self, X):
        '''
        Calculate probabilities  using learned logistic regression parameters (w, b)

        Arguments:

        X -- data of size (number of examples * num_features)

        Returns:
        numpy array (vector) containing predictions for the outcomes
        '''

        m = X.shape[0]
        X = X.toarray()

        if len(self.w) != 0:
            w = self.w
            b = self.b
        else:
            dim = np.shape(X)[1]
            w, b = self.initialize_weight(dim)
        H = self.hypothesis(w, X, b)
        return H

    def iterate_minibatches(self, X_train, Y_train, batchsize=None, shuffle=False):
        '''
        Iterator that allows generate batches shuffeled or not with given size and feed them to the model.

        Arguments:

        X_train -- whole matrix of train data
        Y_train -- whole matrix of groun truth
        batchsize -- size of batches (if nothing is set then the whole data)
        shuffle -- indicator to shuffle the data (by default no shuffle)

        Returns:
        Iterator with batches
        '''
        assert X_train.shape[0] == len(Y_train)
        if shuffle:
            indices_shuffle = np.random.permutation(Y_train.shape[0])
        else:
            indices = range(Y_train.shape[0])
        if batchsize != None:
            for start_idx in range(0, X_train.shape[0] - batchsize + 1, batchsize):
                if shuffle:
                    excerpt = indices_shuffle[start_idx:start_idx + batchsize]
                else:

                    excerpt = indices[start_idx:start_idx + batchsize]

                yield X_train[excerpt].toarray(), Y_train[excerpt]
        else:
            batchsize = len(Y_train)
            yield X_train.toarray(), Y_train.reshape(batchsize, 1)