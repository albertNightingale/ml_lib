
from typing import Union

import math
import numpy as np

class Node: 
  def __init__(self, node_name, init_val, layer_number = -1):
    self.node_name = node_name
    self.layer_number = layer_number
    self.val = init_val
    self.loss_gradient = 0

  def __str__(self):
    return str(self.node_name)


class NeuralNetwork:
  def __init__(self, node_map: dict[str, Node], input_names: list[str], output_name: str, weights: np.ndarray, num_layers, num_nodes, num_input_nodes, method="sigmoid", epochs=100):
    self.nodes = node_map
    self.input_names = input_names
    self.output_name = output_name
    self.weights = weights
    
    self.num_layers = num_layers # number of layers
    self.num_nodes = num_nodes # total number of nodes
    self.num_input_nodes = num_input_nodes # number of input nodes
    
    self.method = method
    self.epochs = epochs
    self.gamma_0 = 0.1

  def _initialize_weights(self, weight):
    for node_weight_pair in self.weights:
      node_weight_pair[2] = weight

  def learning_rate_schedule(self, T):
    return self.gamma_0 / (1 + T)

  def _sigmoid(self, x):
    return 1 / (1 + math.exp(-x))

  def _sigmoid_derivative(self, val):
    sigmoid = self._sigmoid(val)
    return sigmoid * (1 - sigmoid)

  # forward pass
  def forward_pass(self, x):    
    # input nodes
    for idx, input_name in enumerate(self.input_names):
      self.nodes[input_name].val = x[idx]

    # inner nodes
    for layer_number in range(1, self.num_layers, 1):
      print("forward_pass: layer_number: ", layer_number)
      for _name in self.nodes: 
        if self.nodes[_name].layer_number != layer_number: # not the same layer number
          continue
        self.nodes[_name].val = self._compute_node(_name)
    
    # output node
    output_node = self.nodes[self.output_name]
    output_node.val = self._linear_combination(self.output_name)

  # compute the value of node_name by linear combination the values of the children
  def _compute_node(self, node_name: str) -> float:
    lc = self._linear_combination(node_name)

    if type(lc) == bool: # not computing the sigma because there is no children
      return 1

    sigma_lc = self._sigmoid(lc)
    return sigma_lc

  def _linear_combination(self, node_name: str) -> Union[float, bool]:
    children_weights = self.weights[self.weights[:,0] == node_name] # get its childrens
    print("childrens of {}: ".format(node_name))
    print(children_weights)
    children_val: np.ndarray = np.array([self.nodes[child_name].val for child_name in children_weights[:,1]]) # map children values
    weights: np.ndarray = children_weights[:,2].astype(float) # map children weights
    print("children_val: ", children_val)
    print("weights: ", weights)

    # if there is no children, return false
    if children_val.shape[0] == 0:
      return False

    _dot = np.dot(children_val, weights)
    print("resulting dot: {}".format(_dot))
    print()
    return _dot
  
  # compute the gradient of the weight under node with node_name
  def _gradient_weight(self, node_name, _gradients, apply_activation: bool):
    _node = self.nodes[node_name]
    _filtered_indices = np.argwhere(self.weights[:,0] == node_name).flatten() # filter the weights between node and it's children

    activation_derivative = 0
    if apply_activation:
      activation_derivative = self._sigmoid_derivative(self._linear_combination(node_name))
      print("activation derivative of {} is {}".format(node_name, activation_derivative))

    for i in _filtered_indices:
      _child_name = self.weights[i][1]
      _child_val = self.nodes[_child_name].val
      
      _grad = _node.loss_gradient * _child_val
      if apply_activation:
        _grad *= activation_derivative
      _gradients[i] = _grad

  # gradient computation algorithm
  # assume the graph is already initialized using the forward pass with x
  def compute_gradient(self, y_star):
    # initialize an empty array of weights for gradient of each weight
    _gradients = np.zeros(self.weights.shape[0])
    out_node = self.nodes[self.output_name]

    # compute the gradient L / gradient y
    out_node.loss_gradient = out_node.val - y_star
    self._gradient_weight(self.output_name, _gradients, False)
    
    # compute the gradient L / gradient x
    for layer_number in range(self.num_layers - 1, 0, -1):
      print("_compute_gradient: layer_number: ", layer_number)
      for _name in self.nodes:
        _node = self.nodes[_name] 
        if _node.layer_number != layer_number: # not the same layer number
          continue
        
        print("processing node: ", _name)
        # compute the gradient of the node
        parents_weights = self.weights[self.weights[:,1] == _name]
        
        _parent_loss_grads = np.array([self.nodes[parent].loss_gradient for parent in parents_weights[:,0]]).flatten() # map parent's loss gradients
        _parent_weights = parents_weights[:, 2].astype(float)

        _node.loss_gradient = np.dot(_parent_loss_grads, _parent_weights)
        print("dot product parent_loss_grad {} with _parent_weights {} result in node gloss: {}\n".format(_parent_loss_grads, _parent_weights, _node.loss_gradient))

        # compute the gradient of the weight
        self._gradient_weight(_name, _gradients, True)

    return _gradients


  # def fit(self, X, Y, initialize_weights=True):
  #   if initialize_weights:
  #     self._initialize_weights(0.1)

  #   for T in self.epoch:
  #     for i in range(len(X)):
  #       x = X[i]
  #       # forward pass to compute the value of all nodes for x = X[i]
  #       self.forward_pass(x)
  #       # for each weight, compute the gradient and update the weight
  #       for node_weight_pair in self.weights:      
  #         # compute the gradient of the loss function on weights
  #         gradient = self._compute_gradient(node_weight_pair, Y[i])
  #         # update weights
  #         node_weight_pair[2] = node_weight_pair[2] - self.learning_rate_schedule(T) * gradient
  #   return self.weights

