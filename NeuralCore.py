import math
import json

def sigmoid(x):
    if x < -700: return 0.0
    if x > 700: return 1.0
    return 1 / (1 + math.exp(-x))

def derivsig(y):
    return y * (1 - y)

def softmax(x):
    max_x = max(x)  # for stability
    exps = [math.exp(i - max_x) for i in x]
    sum_exps = sum(exps)
    return [j / sum_exps for j in exps]

class NeuralNetwork:
    def __init__(self, weights, config_path="E:\\py\\Config.json"):
        self._w1, self._w2, self._w3 = weights
        with open(config_path, "r") as f:
            layer_sizes = json.load(f)
        self.input_size = layer_sizes[0] // int(math.sqrt(layer_sizes[1]))
        self.hidden1_size = int(math.sqrt(layer_sizes[1]))
        self.hidden2_size = int(math.sqrt(layer_sizes[1]))
        self.output_size = layer_sizes[2] // int(math.sqrt(layer_sizes[1]))

        self.hidden1 = [0.0] * self.hidden1_size
        self.hidden2 = [0.0] * self.hidden2_size
        self.out = [0.0] * self.output_size

    def forward(self, inp):
        for j in range(self.hidden1_size):
            s = sum(inp[k] * self._w1[j * self.input_size + k] for k in range(self.input_size))
            self.hidden1[j] = sigmoid(s)

        for j in range(self.hidden2_size):
            s = sum(self.hidden1[k] * self._w2[j * self.hidden1_size + k] for k in range(self.hidden1_size))
            self.hidden2[j] = sigmoid(s)

        raw = []
        for j in range(self.output_size):
            s = sum(self.hidden2[k] * self._w3[j * self.hidden2_size + k] for k in range(self.hidden2_size))
            raw.append(s)
        self.out = softmax(raw)


        return self.out

    def backprop(self, inp, label, lr):
        ideal = [0.0] * self.output_size
        ideal[label] = 1.0

        d_out = [self.out[j] - ideal[j] for j in range(self.output_size)]

        d_2nd = [0.0] * self.hidden2_size
        for i2 in range(self.hidden2_size):
            delta = 0.0
            for j in range(self.output_size):
                delta += d_out[j] * self._w3[j * self.hidden2_size + i2]
                self._w3[j * self.hidden2_size + i2] -= lr * d_out[j] * self.hidden2[i2]
            d_2nd[i2] = delta * derivsig(self.hidden2[i2])

        d_1st = [0.0] * self.hidden1_size
        for i1 in range(self.hidden1_size):
            delta = 0.0
            for j in range(self.hidden2_size):
                delta += d_2nd[j] * self._w2[j * self.hidden1_size + i1]
                self._w2[j * self.hidden1_size + i1] -= lr * d_2nd[j] * self.hidden1[i1]
            d_1st[i1] = delta * derivsig(self.hidden1[i1])

        for j in range(self.hidden1_size):
            for k in range(self.input_size):
                self._w1[j * self.input_size + k] -= lr * d_1st[j] * inp[k]
        
    def predict(self):
        return self.out.index(max(self.out))

import Variables
NeuralNetwork(Variables.Weights().weights)
