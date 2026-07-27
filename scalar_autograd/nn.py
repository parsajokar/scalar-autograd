import json
import random
from scalar_autograd import Value


class Neuron:
    def __init__(self, nin, nonlin=True):
        self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)]
        self.b = Value(0)
        self.nonlin = nonlin

    def __call__(self, x):
        act = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
        return act.relu() if self.nonlin else act

    def parameters(self):
        return self.w + [self.b]


class Layer:
    def __init__(self, nin, nout, nonlin=True):
        self.neurons = [Neuron(nin, nonlin=nonlin) for _ in range(nout)]

    def __call__(self, x):
        out = [n(x) for n in self.neurons]
        return out[0] if len(out) == 1 else out

    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]


class MultilayerPerceptron:
    def __init__(self, nin, nouts):
        self.nin = nin
        self.nouts = list(nouts)
        sz = [nin] + nouts
        self.layers = [
            Layer(sz[i], sz[i + 1], nonlin=i != len(nouts) - 1)
            for i in range(len(nouts))
        ]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

    def predict(self, x):
        out = self(x)
        if isinstance(out, list):
            return max(range(len(out)), key=lambda i: out[i].data)
        return out.data

    def save(self, path):
        state = {
            "nin": self.nin,
            "nouts": self.nouts,
            "parameters": [p.data for p in self.parameters()],
        }
        with open(path, "w") as f:
            json.dump(state, f)

    @classmethod
    def load(cls, path):
        with open(path) as f:
            state = json.load(f)
        model = cls(state["nin"], state["nouts"])
        for p, data in zip(model.parameters(), state["parameters"]):
            p.data = data
        return model
