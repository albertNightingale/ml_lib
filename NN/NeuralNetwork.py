
import numpy as np

class Node: 
  def __init__(self, node_name, init_val, layer_number = -1):
    self.node_name = node_name
    self.val = init_val
    self.layer_number = layer_number

class NeuralNetwork:
  def __init__(self, input_nodes: list[Node], inner_nodes: list[Node], output_node: Node, point_to, num_layers, num_nodes, num_input_nodes, method="sigmoid", epochs=100):
    self.input_nodes = input_nodes
    self.inner_nodes = inner_nodes
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

  def _sigmoid_derivative(self, _weights, _xs):
    sigmoid = self._sigmoid(np.dot(_xs, _weights))
    return sigmoid * (1 - sigmoid)

  ###################
  # path finding algorithm
  def _find_paths(self, data, start, end):
    return self._dfs(data, end, [start], [])
  # given data, ending point, and path with start point, find all the paths. 
  def _dfs(self, data, end_point, path, paths):
      start_point = path[-1]
      if start_point == end_point:
          paths += [path]
          return paths
      
      if start_point not in data[:,0]: # start has no children        
          return paths
      
      start_point_children = data[data[:,0]==start_point]
      for child in start_point_children:
          temp_path = path + [child[1]]
          paths = self._dfs(data, end_point, temp_path, paths)
      
      return paths

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

  def compute_node(self, node_name):
    related_weights = []
    related_node_values = []

    _filtered_weights = self.weights[self.weights[:,0] == node_name]
    for weight in _filtered_weights:
      related_weights.append(weight[2])
      
      for _node in self.inner_nodes.tolist() + self.input_nodes.tolist():
        if _node.node_name == target_weight[1]:
          target_weight_node = _node
          break

    return related_weights
    
  # gradient computation algorithm
  # assume the graph is already initialized using the forward pass with x
  def _compute_gradient(self, target_weight: list[str], y_star):
    grad_loss_to_weight = 0

    target_weight_node:Node = None
    # find the target weight node object
    for _node in self.inner_nodes.tolist() + self.input_nodes.tolist():
      if _node.node_name == target_weight[1]:
        target_weight_node = _node
        break

    # gradient of target to weight is the target's child
    # target = sigma(target_child_0 * weight_0 + target_child_1 * weight_1 + target_child_2 * weight_2)
    # gradient(target)/gradient(weight_0) = (1-sigma)(sigma)target_child_0
    grad_target_to_weight = target_weight_node.val # target_weight[1]'s value

    target = target_weight[0]
    paths = self._find_paths(self.weights, self.output_node.node_name, target)
    print("found {} paths...".format(len(paths)))
    
    for path in paths:
      g = 0 # use chain rule for the gradient
      parent_node = "L" # the starting parent node is the loss function itself
      for _node in path:
        if _node == self.output_node.node_name:
          y = self.output_node.val
          g = y - y_star
        elif _node == target:
          g *= float(grad_target_to_weight)
        else: # gradient(parent)/gradient(n) in a layer is the weights of the linear function of their parents
          parent_filter = self.weights[self.weights[:,0] == parent_node]
          n_filter = parent_filter[parent_filter[:,1] == _node]
          g *= np.sum(n_filter[:,2].astype(float))
        parent_node = _node
      
      grad_loss_to_weight += g
      # print(grad_loss_to_weight)
    return grad_loss_to_weight


  def fit(self, X, Y, initialize_weights=True):
    if initialize_weights:
      self._initialize_weights(0.1)

    for T in self.epoch:
      for i in range(len(X)):
        x = X[i]
        # forward pass to compute the value of all nodes for x = X[i]
        self.forward_pass(x)
        # for each weight, compute the gradient and update the weight
        for node_weight_pair in self.weights:      
          # compute the gradient of the loss function on weights
          gradient = self._compute_gradient(node_weight_pair, Y[i])
          # update weights
          node_weight_pair[2] = node_weight_pair[2] - self.learning_rate_schedule(T) * gradient
    return self.weights

