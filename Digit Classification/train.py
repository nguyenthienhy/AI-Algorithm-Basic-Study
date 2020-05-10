import load_datasets
from neural_one_hidden_layer import nn_model

X_data , y_data = load_datasets.X_data , load_datasets.y_data

# mỗi một ảnh có shape : 20 * 20 => có 400 units vào , chưa kể bias
# assert(X_data.shape == (400 , 5000))
# mạng neural gồm :
## input : 400
## hidden : 24
## output : 10

X_assess, Y_assess = X_data / 255 , y_data / 255

parameters = nn_model(X_assess, Y_assess, 24 , num_iterations=10000, print_cost=True)

print("W1 = " + str(parameters["W1"]))
print("b1 = " + str(parameters["b1"]))
print("W2 = " + str(parameters["W2"]))
print("b2 = " + str(parameters["b2"]))
