
from typing import Union

import time
import math
import random
import numpy as np

class Node: 
  def __init__(self, node_name, layer_number = -1):
    self.node_name = node_name
    self.layer_number = layer_number

    self.val = 0
    self.loss_gradient = 0

  def __str__(self):
    return str(self.node_name)

class NeuralNetwork:
  def __init__(self, num_layers, num_nodes_per_layer, num_input_nodes, debug=False, method="sigmoid", epoch=100):
    self.num_layers = num_layers # number of layers
    self.num_nodes_per_layer = num_nodes_per_layer # total number of nodes
    self.num_input_nodes = num_input_nodes # number of input nodes
      
    # initialize_nodes
    self.input_names = list()
    self.output_name = ""
    self.node_name_by_layer = dict[int, list]()
    self.nodes: dict[str, Node] = self._initialize_nodes()
    self.weights: np.ndarray = self._initialize_weights()

    self.debug = debug
    self.method = method
    self.epoch = epoch
    self.gamma_0 = 0.1

  def _initialize_nodes(self): 
    nodes: dict[str, Node] = dict()

    for layer in range(self.num_layers + 1):
      self.node_name_by_layer[layer] = list()

    nodes = {"y_1": Node("y_1", self.num_layers)}
    self.node_name_by_layer[self.num_layers].append("y_1")
    self.output_name = "y_1"

    for i in range(self.num_input_nodes):
      input_node_name = "x_{}".format(i)
      self.input_names.append(input_node_name)
      nodes[input_node_name] = Node(input_node_name, 0)
      self.node_name_by_layer[0].append(input_node_name)

    for layer in range(1, self.num_layers, 1):
      for i in range(self.num_nodes_per_layer):
        node_name = "z_{}_{}".format(i, layer)
        nodes[node_name] = Node(node_name, layer)
        self.node_name_by_layer[layer].append(node_name)
    return nodes

  def _initialize_weights(self):
    weights = list()
    for layer in range(self.num_layers):
      for node_name in self.node_name_by_layer[layer]:
        for parent_name in self.node_name_by_layer[layer + 1]:
          weights.append(np.array([parent_name, node_name, random.random()], dtype=object))
    return np.array(weights)
    
  def set_custom_weights(self, weights):
    self.weights = weights

  def learning_rate_schedule(self, T):
    d = 7
    return self.gamma_0 / (1 + self.gamma_0 / d * T)

  # sigmoid function
  def _sigmoid(self, x):
    try:
      return 1 / (1 + math.exp(-x))
    except OverflowError:
      if x < 0:
        return 0
      else:
        return 1

  # sigmoid derivative
  def _sigmoid_derivative(self, val):
    sigmoid = self._sigmoid(val)
    return sigmoid * (1 - sigmoid)

  # linear combination
  def _linear_combination(self, node_name: str) -> Union[float, bool]:
    children_weights = self.weights[self.weights[:,0] == node_name] # get its childrens
    self.debug and print("childrens of {}: \n{}".format(node_name, children_weights))
    children_val: np.ndarray = np.array([self.nodes[child_name].val for child_name in children_weights[:,1]]) # map children values
    weights: np.ndarray = children_weights[:,2].astype(float) # map children weights
    self.debug and print("children_val: ", children_val, ", weights: ", weights)

    # if there is no children, return false
    if children_val.shape[0] == 0:
      return False

    _dot = np.dot(children_val, weights)
    self.debug and print("resulting dot: {}\n".format(_dot))
    return _dot

  # forward pass
  def forward_pass(self, x):    
    # input nodes
    for idx, input_name in enumerate(self.input_names):
      self.nodes[input_name].val = x[idx]

    # inner nodes
    for layer_number in range(1, self.num_layers, 1):
      self.debug and print("forward_pass: layer_number: ", layer_number)
      for _name in self.nodes: 
        if self.nodes[_name].layer_number != layer_number: # not the same layer number
          continue
        self.nodes[_name].val = self._compute_node(_name)
    
    # output node
    val = self._linear_combination(self.output_name)
    self.nodes[self.output_name].val = val
    return val

  # compute the value of node_name by linear combination the values of the children
  def _compute_node(self, node_name: str) -> float:
    lc = self._linear_combination(node_name)

    if type(lc) == bool: # not computing the sigma because there is no children
      return 1

    sigma_lc = self._sigmoid(lc)
    return sigma_lc
  
  # compute the gradient of the weight under node with node_name
  def _gradient_weight(self, node_name, _gradients, apply_activation: bool):
    _node = self.nodes[node_name]
    _filtered_indices = np.argwhere(self.weights[:,0] == node_name).flatten() # filter the weights between node and it's children

    activation_derivative = 0
    if apply_activation:
      activation_derivative = self._sigmoid_derivative(self._linear_combination(node_name))
      self.debug and print("activation derivative of {} is {}".format(node_name, activation_derivative))

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
      self.debug and print("_compute_gradient: layer_number: ", layer_number)
      for _name in self.node_name_by_layer[layer_number]:
        _node = self.nodes[_name] 
        
        self.debug and print("processing node: ", _name)
        # compute the gradient of the node
        parents_weights = self.weights[self.weights[:,1] == _name]
        
        _parent_loss_grads = np.array([self.nodes[parent].loss_gradient for parent in parents_weights[:,0]]).flatten() # map parent's loss gradients
        _parent_weights = parents_weights[:, 2].astype(float)

        _node.loss_gradient = np.dot(_parent_loss_grads, _parent_weights)
        self.debug and print("dot product parent_loss_grad {} with _parent_weights {} result in node gloss: {}\n".format(_parent_loss_grads, _parent_weights, _node.loss_gradient))

        # compute the gradient of the weight
        self._gradient_weight(_name, _gradients, True)

    return _gradients


  def fit(self, X: np.ndarray, Y: np.ndarray):
    for T in range(self.epoch):
      # start_time = time.time()
      # print("Iteration :", T)

      merged_data = np.column_stack((X, Y))
      np.random.shuffle(merged_data)
      X, Y = merged_data[:, :-1], merged_data[:, -1]
      for i in range(len(X)):
        x = X[i]
        # forward pass to compute the value of all nodes for x = X[i]
        self.forward_pass(x)
        gradient_weights = self.compute_gradient(Y[i])
        for j in range(len(gradient_weights)):
          self.weights[j][2] -= self.learning_rate_schedule(T) * gradient_weights[j]

      # print("end T {} took {} seconds ---".format(T, time.time() - start_time))
        
  def predict(self, X: np.ndarray):
    Y_hat = np.zeros(X.shape[0])
    for i, _x in enumerate(X):
      Y_hat[i] = self.forward_pass(_x) # self._sigmoid(self.forward_pass(_x))
    
    return Y_hat
