import math
import random

class Neuron:
    def __init__(self, nin, final):
        self.weights = [Value(random.uniform(-1,1)) for _ in range(nin)]
        self.bias = Value(random.uniform(-1,1))
        self.final = final

    def __call__(self, inpts):
        out = sum([weight * inpt for weight, inpt in zip(self.weights, inpts)], self.bias)
        if not self.final:
            out = out.tanh()
        return out

    def parameters(self):
        out = self.weights + [self.bias]
        return out

class Layer:
    def __init__(self, nin, nout, final):
        self.neurons = [Neuron(nin, final) for _ in range(nout)]

    def __call__(self, inpt):
        out = [neuron(inpt) for neuron in self.neurons]
        return out[0] if len(out) == 1 else out

    def parameters(self):
        out = [p for neuron in self.neurons for p in neuron.parameters()]
        return out

class NLP:
    def __init__(self, nin, layers):
        layers_i = [nin] + layers
        self.layers = [Layer(layers_i[i], layers_i[i+1], False) if i != len(layers)-1 else Layer(layers_i[i], layers_i[i+1], True) for i in range(len(layers))]

    def __call__(self, inpts):
        x = inpts
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        out = [p for layer in self.layers for p in layer.parameters()]
        return out
