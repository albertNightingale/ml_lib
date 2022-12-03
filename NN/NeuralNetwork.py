
import numpy as np

class Node: 
  def __init__(self, node_name, init_val, layer_number = -1):
    self.node_name = node_name
    self.layer_number = layer_number
    self.val = init_val
    self.loss_gradient = 0

class NeuralNetwork:
  def __init__(self, node_map: dict[str, Node], output_node: Node, point_to: np.ndarray, num_layers, num_nodes, num_input_nodes, method="sigmoid", epochs=100):
    self.input_nodes = node_map
    self.output_node = output_node
    self.weights = point_to
    
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
    return 1 / (1 + np.exp(-x))

  def _sigmoid_derivative(self, val):
    sigmoid = self._sigmoid(val)
    return sigmoid * (1 - sigmoid)

  # forward pass
  def forward_pass(self, x):
    for i in range(self.num_input_nodes):
      self.input_nodes[i].val = x[i] 
    
    q = self.input_nodes.tolist()

    # inverse bfs to fill out the value of all nodes
    while len(q) > 0:
      _node: Node = q.pop(0)
      _links = self.weights[self.weights[:,1] == _node.node_name]
      for _node_link in _links:
        parent_name, w = _node_link[0], float(_node_link[2])
        if self.output_node.node_name == parent_name:
          self.output_node.val += _node.val * w
        
        # search in inner nodes
        for _inner_node in self.inner_nodes:
          if _inner_node.node_name == parent_name:
            _inner_node.val += _node.val * w
            q.append(_inner_node)

  # compute the value of node_name by linear combination the values of the children
  def _compute_node(self, node_name):
    _matching_weights = self.weights[self.weights[:,0] == node_name] 
    children_val: np.ndarray = np.array([self.input_nodes[child_name].val for child_name in _matching_weights[:,1]])
    weights: np.ndarray = _matching_weights[:,2].astype(float)
  
    return np.dot(children_val, weights)
    
  # gradient computation algorithm
  # assume the graph is already initialized using the forward pass with x
  def _compute_gradient(self, y_node: Node, y_star):
    # initialize an empty array of weights for gradient of each weight
    _gradients = np.zeros(self.weights.shape[0])

    # compute the gradient L / gradient y
    y_node.loss_gradient = y_node.val - y_star
    y_node_idx = self.weights.argwhere(self.weights[:,0] == y_node.node_name)
    # iterate all links and compute the gradient of each weight
    for i in y_node_idx:
      _child_name = self.weights[i,1]
      _child_val = self.input_nodes[_child_name].val
      _gradients[i] = y_node.loss_gradient * _child_val
    
    
    # compute the gradient L / gradient x
    for i in range(self.num_layers - 2, -1, -1):
      for j in range(self.num_nodes[i]):
        _node_name = f"layer_{i}_node_{j}"
        _node = self.input_nodes[_node_name]
        _node_idx = self.weights.argwhere(self.weights[:,0] == _node_name)
        _node.loss_gradient = 0
        for k in _node_idx:
          _child_name = self.weights[k,1]
          _child_val = self.input_nodes[_child_name].val
          _child_loss_gradient = self.input_nodes[_child_name].loss_gradient
          _gradients[k] = _child_loss_gradient * _child_val
          _node.loss_gradient += _child_loss_gradient * self.weights[k,2]
        
        _node.loss_gradient *= self._sigmoid_derivative(_node.val)    

    return 


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

