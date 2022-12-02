import numpy as np

from NN.NeuralNetwork import NeuralNetwork
from NN.NeuralNetwork import Node

input_nodes = np.array([
  Node("x_0", 1, layer_number=0),
  Node("x_1", 1, layer_number=0), 
  Node("x_2", 1, layer_number=0)
])

inner_nodes = np.array([
  Node("z_0_1", 0, layer_number=1),
  Node("z_1_1", 0, layer_number=1),
  Node("z_2_1", 0, layer_number=1),
  Node("z_0_2", 0, layer_number=2), 
  Node("z_1_2", 0, layer_number=2), 
  Node("z_2_2", 0, layer_number=2)
])

outer_node = Node("y_1", 0, layer_number=3)

point_to = np.array([
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

# def _read(filename):
#   result = np.genfromtxt(filename, delimiter=',')
#   _r, _c = result.shape
#   return result[:, :_c-1], result[:, _c-1]

# def _process_boolean_labels(Y):
#   return np.where(Y == 0, -1, Y)

def main():
  X = np.array([1, 1, 1])
  Y = np.array([1])
  # create neural network
  nn = NeuralNetwork(input_nodes, inner_nodes, outer_node, point_to, num_layers=3, num_nodes=10, num_input_nodes=3)
  nn.forward_pass(X)
  for weight in nn.weights:
    print("weight pointing from {} to {} is {}".format(weight[1], weight[0], weight[2]))
    g = nn._compute_gradient(weight, Y)
    print("gradient is {}".format(g))
