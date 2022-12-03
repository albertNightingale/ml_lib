import numpy as np

from NN.NeuralNetwork import NeuralNetwork
from NN.NeuralNetwork import Node

output_node = Node("y_1", 0, layer_number=3)
node_map = {
  "x_0": Node("x_0", 1, layer_number=0),
  "x_1": Node("x_1", 1, layer_number=0), 
  "x_2": Node("x_2", 1, layer_number=0),
  "z_0_1": Node("z_0_1", 0, layer_number=1),
  "z_1_1": Node("z_1_1", 0, layer_number=1),
  "z_2_1": Node("z_2_1", 0, layer_number=1),
  "z_0_2": Node("z_0_2", 0, layer_number=2), 
  "z_1_2": Node("z_1_2", 0, layer_number=2), 
  "z_2_2": Node("z_2_2", 0, layer_number=2),
  "y_1": output_node
}

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

def main():
  X = np.array([1, 1, 1])
  Y = np.array([1])
  # create neural network
  nn = NeuralNetwork(node_map, output_node, point_to, num_layers=3, num_nodes=10, num_input_nodes=3)
  nn.forward_pass(X)
  for weight in nn.weights:
    print("weight pointing from {} to {} is {}".format(weight[1], weight[0], weight[2]))
    g = nn._compute_gradient(weight, Y)
    print("gradient is {}".format(g))
