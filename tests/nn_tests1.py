import numpy as np

from NN.NeuralNetwork import NeuralNetwork
from NN.NeuralNetwork import Node

# node_map = {
#   "x_0": Node("x_0", layer_number=0),
#   "x_1": Node("x_1", layer_number=0), 
#   "x_2": Node("x_2", layer_number=0),
#   "z_0_1": Node("z_0_1", layer_number=1),
#   "z_1_1": Node("z_1_1", layer_number=1),
#   "z_2_1": Node("z_2_1", layer_number=1),
#   "z_0_2": Node("z_0_2", layer_number=2), 
#   "z_1_2": Node("z_1_2", layer_number=2), 
#   "z_2_2": Node("z_2_2", layer_number=2),
#   "y_1": Node("y_1", layer_number=3)
# }

w = np.array([
  ["y_1", "z_0_2", -1], 
  ["y_1", "z_1_2", 2], 
  ["y_1", "z_2_2", -1.5], 
  ["z_1_2", "z_0_1", -1], 
  ["z_2_2", "z_0_1", 1], 
  ["z_1_2", "z_1_1", -2], 
  ["z_2_2", "z_1_1", 2], 
  ["z_1_2", "z_2_1", -3], 
  ["z_2_2", "z_2_1", 3], 
  ["z_1_1", "x_0", -1], 
  ["z_2_1", "x_0", 1], 
  ["z_1_1", "x_1", -2], 
  ["z_2_1", "x_1", 2], 
  ["z_1_1", "x_2", -3], 
  ["z_2_1", "x_2", 3]
])

def main():
  # format np printing
  float_formatter = "{:.4f}".format
  np.set_printoptions(formatter={'float_kind':float_formatter})
  
  X = np.array([1, 1, 1])
  Y = np.array([1])
  # create neural network
  nn = NeuralNetwork(3, 3, 3)
  nn.set_custom_weights(w)
  nn.forward_pass(X)
  
  print("Node value after forward pass: ")
  for _node_name in nn.nodes:
    _node = nn.nodes[_node_name]
    print("Node {} value is {:4f}".format(_node.node_name, _node.val))

  print()

  gradient_weights = nn.compute_gradient(Y[0])
  
  print()
  print("SUMMARY: gradient_loss of each node:")
  for _node_name in nn.nodes:
    _node = nn.nodes[_node_name]
    print("Node {} gloss is {:4f}".format(_node.node_name, _node.loss_gradient))
  print()

  print("Gradient over Weights")
  for i in range(nn.weights.shape[0]):
    weight = nn.weights[i]
    grad: float = gradient_weights[i]
    print("gradient weight pointing from {} to {} is {} ----- : {:4f}".format(weight[1], weight[0], weight[2], grad))


