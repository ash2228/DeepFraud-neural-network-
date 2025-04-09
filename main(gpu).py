import random
import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tqdm import trange

df = pd.read_csv("creditcard.csv")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
X = df.drop(columns=["Class"])
y = df["Class"]

scaler = StandardScaler()
X[["Amount", "Time"]] = scaler.fit_transform(X[["Amount", "Time"]])

X_train, X_test, y_train, y_test = train_test_split(
    X.values.tolist(), y.values.tolist(), test_size=0.2, stratify=y
)

X_train = X_train[100:200]
y_train = y_train[100:200]

def Sigmoid(x):
    return 1 / (1 + torch.exp(-x))

def Sigmoid_derivative(x):
    sx = Sigmoid(x)
    return sx * (1 - sx)

def binary_cross_entropy(y_true, y_pred):
    epsilon = 1e-15
    y_pred = max(min(y_pred, 1 - epsilon), epsilon)
    return -(y_true * torch.log(y_pred) + (1 - y_true) * torch.log(1 - y_pred))

class Neuron:
    def __init__(self, input_size):
        self.weights = torch.randn(input_size, device=device, dtype=torch.float32)
        self.bias = random.uniform(0, 1)
        self.z = 0
        self.output = 0
        self.input = None

    def forward_pass(self, input_val):
        self.input = torch.tensor(input_val, dtype=torch.float32, device=device)
        self.z = torch.dot(self.weights, self.input) + self.bias
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
        dL_dinputs = torch.zeros(len(self.neurons[0].input), device=device)
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
            avg_loss = total_loss / len(self.inputs)
            print(f"Epoch {epoch + 1}, Loss: {avg_loss:.6f}")


neural_network = NeuralNetwork(X_train, y_train, hidden_layer_sizes=[3,3])
neural_network.train(iters=200)

for sample_input, label in zip(X_test[:100], y_test[:100]):
    output = sample_input
    for layer in neural_network.neural_layers:
        output = layer.proceed_pass(output)
    print(f"Real: {label}, Predicted: {output[0]:.4f}")