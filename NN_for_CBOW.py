import numpy as np
from numpy import savetxt
from datetime import datetime 

class Layer:

    def __init__(self):
        pass
    
    def forward(self, input):
        return input

    def backward(self, input, grad_output):
        num_units = input.shape[1]
        
        d_layer_d_input = np.eye(num_units)
        
        return np.dot(grad_output, d_layer_d_input)
    
class ReLU(Layer):
    def __init__(self):
        pass
    
    def forward(self, input):
        return input*(input>0)
    
    def backward(self, input, grad_output):
        relu_grad = input > 0
        return grad_output*relu_grad

class Embeding(Layer):
    def __init__(self, input_units, output_units, learning_rate = None, initialization = None, weights = None):
        self.learning_rate = learning_rate
        
        if initialization == "custom":
            self.weights = weights
        
        elif initialization == "xavier":
            self.weights = np.random.normal(loc=0.0, scale = np.sqrt(2/(input_units+output_units)), 
                                        size = (input_units,output_units))
        else:
            self.weights = np.random.normal(size = (input_units,output_units))
    
    def forward(self, input):
        return input.dot(self.weights)
    
    def bacward(self, input):
        
        return input
    
    def get_weights(self,input,grad_output):
        grad_input = self.weights
        
        grad_weights = input
        
        self.weights = self.weights - self.learning_rate * grad_weights
        
        return grad_input

class Dense(Layer):
    def __init__(self, input_units, output_units, learning_rate = None, initialization = None, weights = None):

        self.learning_rate = learning_rate
        
        if initialization == "custom":
            self.weights = weights
        
        elif initialization == "xavier":
            self.weights = np.random.normal(loc=0.0, scale = np.sqrt(2/(input_units+output_units)), 
                                        size = (input_units,output_units))
        else:
            self.weights = np.random.normal(size = (input_units,output_units))
            
        # self.biases = np.zeros(output_units)
        
    def forward(self,input):
        """Forward propagetion"""
        return input.dot(self.weights)
    # +self.biases
    
    def backward(self,input,grad_output):
        """Back propagetion, compute new weights and biases"""
        grad_input = grad_output.dot(self.weights.T)

        grad_weights = input.T.dot(grad_output)
        # grad_biases = np.sum(grad_output,axis=0)
        
        self.weights = self.weights - self.learning_rate * grad_weights
        # self.biases = self.biases - self.learning_rate * grad_biases
        
        return grad_input
    
    def get_weights(self):
        
        return self.weights

def softmax_crossentropy_with_logits(logits,reference_answers):
    """Compute crossentropy from logits[batch,n_classes] and ids of correct answers"""
    logits_for_answers = logits[np.arange(len(logits)),reference_answers]
    
    xentropy = - logits_for_answers + np.log(np.sum(np.exp(logits),axis=-1))
    
    return xentropy

def grad_softmax_crossentropy_with_logits(logits,reference_answers):
    """Compute crossentropy gradient from logits[batch,n_classes] and ids of correct answers"""
    ones_for_answers = np.zeros_like(logits)
    
    ones_for_answers[np.arange(len(logits)),reference_answers] = 1
    
    softmax = np.exp(logits) / np.exp(logits).sum(axis=-1,keepdims=True)
    
    return (- ones_for_answers + softmax) / logits.shape[0]

def forward(network, X):
    activations = []
    input = X

    for l in network:
        activations.append(l.forward(input))
        # Updating input to last layer output
        input = activations[-1]
        
    return activations

def predict(network,X):

    logits = forward(network,X)[-1]
    return logits.argmax(axis=-1)

def train(network,X,y):

    # Get the layer activations
    layer_activations = forward(network,X)
    layer_inputs = [X]+layer_activations  #layer_input[i] is an input for network[i]
    logits = layer_activations[-1]
    
    # Compute the loss and the initial gradient
    y = np.argmax(y)
    loss = softmax_crossentropy_with_logits(logits,y)
    loss_grad = grad_softmax_crossentropy_with_logits(logits,y)
    
    # Reverse propogation as this is backprop
    for layer_index in range(len(network))[::-1]:
        layer = network[layer_index]
        
        loss_grad = layer.backward(layer_inputs[layer_index],loss_grad) 
    
        
    return np.mean(loss)

def iterate_minibatches(inputs, targets, batchsize = None, shuffle=False):
    assert inputs.shape[0] == targets.shape[0]
    if shuffle:
        indices = np.random.permutation(inputs.shape[0])
    
    if batchsize != None:
        for start_idx in range(0, inputs.shape[0] - batchsize + 1, batchsize):
            if shuffle:
                excerpt = indices[start_idx:start_idx + batchsize]
            else:
                excerpt = slice(start_idx, start_idx + batchsize)
            yield inputs[excerpt].toarray(), targets[excerpt].toarray()
    else:
        yield inputs.toarray(), targets.toarray()
        
def train_network(X_train, y_train, hidden_neurons, num_epoch, learning_rate, batchsize = None, initialization = None, weights_1 = None, weights_2 = None):
    num_input, num_output = X_train.shape[1], y_train.shape[1]
    network = []
    if initialization != None:
        network.append(Dense(num_input,hidden_neurons, learning_rate, initialization = initialization,weights = weights_1))
        # network.append(ReLU())
        network.append(Dense(hidden_neurons,num_output, learning_rate, initialization = initialization,weights = weights_2))
    else:
        network.append(Dense(num_input,hidden_neurons, learning_rate))
        # network.append(ReLU())
        network.append(Dense(hidden_neurons,num_output, learning_rate))
    results = []
    time_elapsed_list = []

    for epoch in range(num_epoch):
        start_time = datetime.now() 
        passes = 0
        for x_batch,y_batch in iterate_minibatches(X_train, y_train, batchsize = batchsize):
            loss = train(network,x_batch,y_batch)
            # print(loss)
            passes += batchsize
            if passes % 100000 == 0:
                time_elapsed = datetime.now() - start_time
                print("done {:.1%} of epoch - {}, loss: {} \ntime for 100000 samples: {}".format(passes / X_train.shape[0],epoch, loss ,time_elapsed))
        results.append([epoch, loss])
        weights_1 = network[-2].get_weights()
        weights_2 = network[-1].get_weights()
        savetxt('weights_1.csv', weights_1, delimiter=',')
        savetxt('weights_2.csv', weights_2, delimiter=',')
        print("done epoch: {}, loss: {} \n weights saved".format(epoch, loss))
    weights_1 = network[-2].get_weights()
    weights_2 = network[-1].get_weights()
    
    return results,weights_1,weights_2

