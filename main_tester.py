from tests.svm_test2 import main as test2
from tests.svm_test1 import main as test1
from tests.svm_test3 import main as test3

print("primal SVM:")
test1()

print("dual SVM:")
test2()

# print("Gaussian SVM:")
# test3()