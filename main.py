import numpy as np
import random

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("creditcard.csv")

X = df.drop(columns=["Class"])
y = df["Class"]

scaler = StandardScaler()
X[["Amount", "Time"]] = scaler.fit_transform(X[["Amount", "Time"]])

X_train, X_test, y_train, y_test = train_test_split(
    X.values.tolist(), y.values.tolist(), test_size=0.2, stratify=y
)

def Sigmoid(x):
    return 1 / (1 + np.exp(-x))

def Sigmoid_derivative(x):
    sx = Sigmoid(x)
    return sx * (1 - sx)

def binary_cross_entropy(y_true, y_pred):
    epsilon = 1e-15
    y_pred = max(min(y_pred, 1 - epsilon), epsilon)
    return -(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

class Neuron:
    def __init__(self, input_size):
        self.weights = np.random.uniform(-1, 1, input_size)
        self.bias = random.uniform(0, 1)
        self.z = 0
        self.output = 0
        self.input = None

    def forward_pass(self, input_val):
        self.input = np.array(input_val)
        self.z = np.dot(self.weights, self.input) + self.bias
        self.output = Sigmoid(self.z)
        return self.output

    def backward_pass(self, dL_dout, learning_rate):
        dz = dL_dout * Sigmoid_derivative(self.z)
        dw = dz * self.input
        db = dz

        self.weights -= learning_rate * dw
        self.bias -= learning_rate * db

        return dz * self.weights 

class NeuralLayer:
    def __init__(self, input_size, neuron_count):
        self.neurons = [Neuron(input_size) for _ in range(neuron_count)]

    def proceed_pass(self, input_val):
        return [neuron.forward_pass(input_val) for neuron in self.neurons]

    def backward_pass(self, dL_douts, learning_rate):
        dL_dinputs = np.zeros(len(self.neurons[0].input))
        for neuron, dL_dout in zip(self.neurons, dL_douts):
            dL_dinput = neuron.backward_pass(dL_dout, learning_rate)
            dL_dinputs += dL_dinput
        return dL_dinputs

class NeuralNetwork:
    def __init__(self, x, y, hidden_layer_sizes):
        self.inputs = x  
        self.targets = y  

        input_size = len(x[0]) 
        layer_sizes = [input_size] + hidden_layer_sizes + [1]

        self.neural_layers = []
        for i in range(len(layer_sizes) - 1):
            self.neural_layers.append(
                NeuralLayer(layer_sizes[i], layer_sizes[i + 1])
            )

    def train(self, iters, learning_rate=0.1):
        for epoch in range(iters):
            total_loss = 0
            for sample_input, target in zip(self.inputs, self.targets):
                input_val = sample_input

                for layer in self.neural_layers:
                    input_val = layer.proceed_pass(input_val)
                prediction = input_val[0]
                loss = binary_cross_entropy(target, prediction)
                total_loss += loss

                dL_dpred = prediction - target 


                grad = [dL_dpred]
                for layer in reversed(self.neural_layers):
                    grad = layer.backward_pass(grad, learning_rate)


neural_network = NeuralNetwork(X_train, y_train, hidden_layer_sizes=[16, 8])
neural_network.train(iters=1000)

for sample_input, label in zip(X_test[:10], y_test[:10]):
    output = sample_input
    for layer in neural_network.neural_layers:
        output = layer.proceed_pass(output)
    print(f"Real: {label}, Predicted: {output[0]:.4f}")