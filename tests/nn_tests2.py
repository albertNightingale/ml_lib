
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

def _read(filename):
    result = np.genfromtxt(filename, delimiter=',')
    _r, _c = result.shape
    np.random.shuffle(result)
    return result[:, :_c-1], result[:, _c-1]


def _process_boolean_labels(Y):
    return np.where(Y == 0, -1, Y)

def main():
  # format np printing
  float_formatter = "{:.4f}".format
  np.set_printoptions(formatter={'float_kind':float_formatter})
  
  # get and process data and labels
  X, Y = _read('data/hw5/bank-note/train.csv')
  Y = _process_boolean_labels(Y)


  X_test, Y_test = _read('data/hw5/bank-note/test.csv')
  Y_test = _process_boolean_labels(Y_test)

  widths = [5, 10, 25, 50, 100]

  for width in widths:
    print("start with width: ", width)
    # create neural network
    nn = NeuralNetwork(3, width, 3)
    nn.fit(X, Y)

    # print("Final Weights: ")
    # for w in nn.weights:
    #   print("{} point to {} ----- {}".format(w[1], w[0], w[2]))

    Y_hat_test = nn.predict(X_test)

    # for i in range(Y_test.shape[0]):
    #   print("Y_hat_test: {}, Y_test: {}".format(Y_hat_test[i], Y_test[i]))

    # compute accuracy by comparing Y_hat_test rounded to the nearest integer and Y_test 
    accuracy_test = np.sum(np.round(Y_hat_test) == Y_test) / Y_test.shape[0]
    print("accuracy on testing data: {}".format(accuracy_test))

    Y_train = nn.predict(X)
    # compute accuracy by comparing Y_hat_test rounded to the nearest integer and Y_test 
    accuracy_train = np.sum(np.round(Y_train) == Y) / Y.shape[0]
    print("accuracy on training data: {}".format(accuracy_train))
