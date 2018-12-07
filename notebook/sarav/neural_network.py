from scipy.io import arff
import numpy as np
import pandas as pd
import math

class NeuralNetwork():
    def __init__(self, 
                 input_nodes,
                 hidden_nodes=[],
                 output_nodes=1,
                 batch_size=4,
                 learning_rate=1e-4,
                 momentum=0,
                 threshold=0.5):
        assert(input_nodes >= 1)
        assert(0 <= len(hidden_nodes) <= 10)
        assert(batch_size >= 1)
        
        self.layers = self._init_layers(input_nodes, hidden_nodes, output_nodes)
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.threshold = threshold
        
        self.params_values = self._init_weights()

        
    def _init_layers(self, input_nodes, hidden_nodes, output_nodes):
        layers = []
        layers.append(input_nodes)
        for hidden_layer in hidden_nodes:
            layers.append(hidden_layer)
        layers.append(output_nodes)
        
        return layers
    
    def _init_grads_values(self):
        grads_values = {}
        for idx in range(1, len(self.layers)):
            layer_input_size = self.layers[idx-1]
            layer_output_size = self.layers[idx]
            grads_values['W' + str(idx)] = np.zeros([layer_output_size, layer_input_size])
            grads_values['b' + str(idx)] = np.zeros([layer_output_size])
        return grads_values
        
    def _init_weights(self):
        """
        Initiate weights and bias weights for the neural network
        """
        params_values = {}
        for idx in range(len(self.layers)-1):
            layer_input_size = self.layers[idx]
            layer_output_size = self.layers[idx+1]
            
            # Weight
            params_values['W' + str(idx+1)] = np.random.randn(layer_output_size, layer_input_size) * 0.1
            
            # Bias Weight
            params_values['b' + str(idx+1)] = np.random.randn(layer_output_size) * 0.1
            
        return params_values
    
    
    def _single_layer_feed_forward(self, A_prev, W_curr, b_curr):
        """
        Feed forward for single layer in neural network
        """
        Z_curr = np.dot(W_curr, A_prev) + b_curr
        return self._sigmoid(Z_curr), Z_curr
        
        
    def _full_feed_forward(self, X):
        memory = {}
        A_curr = X
        
        for idx in range(len(self.layers)-1):
            A_prev = A_curr
            
            W_curr = self.params_values['W' + str(idx+1)]
            b_curr = self.params_values['b' + str(idx+1)]
            
            A_curr, Z_curr = self._single_layer_feed_forward(A_prev, W_curr, b_curr)
            
            memory['A' + str(idx)] = A_prev
            memory['Z' + str(idx+1)] = Z_curr
            
        memory['A' + str(len(self.layers)-1)] = A_curr
        return A_curr, memory
        
    
    def _single_layer_backward_prop(self, dA_curr, W_curr, b_curr, Z_curr, A_prev):
        
        dZ_curr = dA_curr * self._sigmoid_backward(Z_curr)
        dW_curr = np.outer(dZ_curr, A_prev)
        db_curr = dZ_curr
        dA_prev = np.dot(W_curr.T, dZ_curr)
        
        return dA_prev, dW_curr, db_curr
    
    
    def _full_backward_prop(self, y_hat, y, memory):
        grads_values = {}
        
        dA_curr = y - y_hat
        
        for layer_idx_prev in range(len(self.layers)-2, -1, -1):
            layer_idx_curr = layer_idx_prev + 1
            
            A_curr = memory['A' + str(layer_idx_curr)]
            A_prev = memory['A' + str(layer_idx_prev)]
            Z_curr = memory['Z' + str(layer_idx_curr)]
            W_curr = self.params_values['W' + str(layer_idx_curr)]
            b_curr = self.params_values['b' + str(layer_idx_curr)]
            
            dA_prev, dW_curr, db_curr = self._single_layer_backward_prop(dA_curr, W_curr, b_curr, Z_curr, A_prev)
            
            grads_values['dW' + str(layer_idx_curr)] = dW_curr
            grads_values['db' + str(layer_idx_curr)] = db_curr
            
        return grads_values
    
    
    def _update(self, grads_values):
        for layer_idx in range(1, len(self.layers)):
            self.params_values['W' + str(layer_idx)] += self.learning_rate * grads_values['dW' + str(layer_idx)] + self.momentum * self.params_values['W' + str(layer_idx)]
            self.params_values['b' + str(layer_idx)] += self.learning_rate * grads_values['db' + str(layer_idx)] + self.momentum * self.params_values['b' + str(layer_idx)]

            
    def _sigmoid(self, weighted_sum):
        return 1/(1+np.exp(-weighted_sum))    
    
    
    def _sigmoid_backward(self, y):
        sigmoid = self._sigmoid(y)
        return sigmoid * (1 - sigmoid)
    
        
    def _calc_error(self, output, target):
        return 0.5 * ((output - target) ** 2)
    
    
    def _threshold(self, x):
        return 1 if x > self.threshold else 0
    
    
    def _calc_accuracy(self, output, target):
        count_correct = 0
        for i in range(len(output)):
            if self._threshold(output[i]) == target:
                count_correct += 1
        return count_correct / len(output)
    
    
    def train(self, X, y, epochs):
        assert(len(X) == len(y))
        cost_history = []
        accuracy_history = []
        
        for epoch_idx in range(epochs):
            epoch_accuracy = 0
            epoch_loss = 0
            n_batch = 0
            for batch_start_idx in range(0, len(X), self.batch_size):
                n_batch += 1
                batch_accuracy = 0
                batch_loss = 0
                n_data_batch = 0
                y_hat, cache = None, None
                batch_grads_values = self._init_grads_values()
                for j in range(batch_start_idx, batch_start_idx + self.batch_size):
                    if j >= len(X):
                        break
                    n_data_batch += 1
                    y_hat, cache = self._full_feed_forward(X[j])
                    batch_grads_values = self._full_backward_prop(y_hat, y[j], cache)               
                    batch_accuracy += self._calc_accuracy(y_hat, y[j])
                    batch_loss += self._calc_error(y_hat, y[j])
                    self._update(batch_grads_values)
                epoch_accuracy += float(batch_accuracy) / float(n_data_batch)
                epoch_loss += float(batch_loss) / float(n_data_batch)
                
            acc = float(epoch_accuracy) / float(n_batch)
            accuracy_history.append(float(epoch_accuracy) / float(n_batch))
            cost = float(epoch_loss) / float(n_batch)
            cost_history.append(float(epoch_loss / n_batch))
            
            print("Epoch {}/{}, Loss={}, Accuracy={}".format(epoch_idx, epochs, cost, acc))
            
        return self.params_values, cost_history, accuracy_history
    
    def predict(self, X):
        y_preds = []
        for x in X:
            y_hat, cache = self._full_feed_forward(x)
            y_preds.append(y_hat)
            
        return y_preds