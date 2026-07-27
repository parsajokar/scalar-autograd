import math


class Value:
    def __init__(self, data):
        self.data = data
        self.grad = 0.0

        self._prev = set()
        self._backward = lambda: None

    def backward(self):
        sorted_nodes = []
        visited = set()

        def topological_sort(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    topological_sort(child)
                sorted_nodes.append(v)

        topological_sort(self)

        self.grad = 1.0
        for v in reversed(sorted_nodes):
            v._backward()

    def __repr__(self):
        return f"Value(data={self.data})"

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(data=self.data + other.data)
        out._prev.add(self)
        out._prev.add(other)

        def _backward():
            self.grad += out.grad
            other.grad += out.grad

        out._backward = _backward

        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(data=self.data * other.data)
        out._prev.add(self)
        out._prev.add(other)

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward

        return out

    def __pow__(self, other):
        assert isinstance(other, (int, float)), "only supports int/float powers"
        out = Value(data=self.data**other)
        out._prev.add(self)

        def _backward():
            self.grad += other * self.data ** (other - 1) * out.grad

        out._backward = _backward

        return out

    def __neg__(self):
        return self * -1

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return other + (-self)

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):
        return self * other**-1

    def __rtruediv__(self, other):
        return other * self**-1

    def relu(self):
        out = Value(0 if self.data < 0 else self.data)
        out._prev.add(self)

        def _backward():
            self.grad += (out.data > 0) * out.grad

        out._backward = _backward

        return out

    def exp(self):
        out = Value(math.exp(self.data))
        out._prev.add(self)

        def _backward():
            self.grad += out.data * out.grad

        out._backward = _backward

        return out

    def log(self):
        out = Value(math.log(self.data))
        out._prev.add(self)

        def _backward():
            self.grad += (1.0 / self.data) * out.grad

        out._backward = _backward

        return out

    def tanh(self):
        out = Value(math.tanh(self.data))
        out._prev.add(self)

        def _backward():
            self.grad += (1 - out.data ** 2) * out.grad

        out._backward = _backward

        return out
