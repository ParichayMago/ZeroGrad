import math
import random

class Neuron:
    def __init__(self, nin):
        self.weights = [Value(random.uniform(-1,1)) for _ in range(nin)]
        self.bias = Value(random.uniform(-1,1))

    def __call__(self, inpts):
        out = sum([weight * inpt for weight, inpt in zip(self.weights, inpts)], self.bias)
        out = out.tanh()
        return out 

    def parameters(self):
        out = self.weights + [self.bias]
        return out

class Layer:
    def __init__(self, nin, nout):
        self.neurons = [Neuron(nin) for _ in range(nout)]

    def __call__(self, inpt):
        out = [neuron(inpt) for neuron in self.neurons]
        return out[0] if len(out) == 1 else out

    def parameters(self):
        out = [p for neuron in self.neurons for p in neuron.parameters()]  
        return out

class NLP:
    def __init__(self, nin, layers):
        layers_i = [nin] + layers
        self.layers = [Layer(layers_i[i], layers_i[i+1]) for i in range(len(layers))]

    def __call__(self, inpts): 
        x = inpts
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        out = [p for layer in self.layers for p in layer.parameters()]
        return out