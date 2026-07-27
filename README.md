# scalar-autograd

A tiny scalar-valued autograd engine — reverse-mode automatic differentiation
over individual `Value` scalars, with a small neural-network library built on
top. Educational, dependency-free, in the spirit of
[micrograd](https://github.com/karpathy/micrograd).

## Usage

Every `Value` records the operations that produced it, so calling `.backward()`
fills in the gradient of the output with respect to every input:

```python
from scalar_autograd import Value

a = Value(2.0)
b = Value(-3.0)
c = (a * b + b).tanh()
c.backward()

print(c.data)   # forward value
print(a.grad)   # d c / d a
print(b.grad)   # d c / d b
```

Supported ops: `+  -  *  /  **`, and `.relu()`, `.exp()`, `.log()`, `.tanh()`.

Build a network with the `nn` primitives:

```python
from scalar_autograd import MultilayerPerceptron

model = MultilayerPerceptron(64, [16, 10])   # 64 inputs -> 16 hidden -> 10 outputs
scores = model([0.0] * 64)                    # list of 10 Values
```

## Examples

A digit classifier trained on the UCI optical-digits dataset
(`examples/optdigits.csv`, bundled in the repo) using softmax cross-entropy loss
and plain SGD:

```bash
python examples/train_model.py    # trains and writes examples/optdigits_model.json
python examples/infer_model.py    # loads the model, reports held-out accuracy
```

Training is pure-Python scalar math, so it takes a few minutes; it reaches
~99% train / ~95% test accuracy.

## Development

The library is pure Python with no dependencies and nothing to install — just
clone the repo and run everything from its root. The examples above work as-is;
to run the tests you only need `pytest`:

```bash
pip install pytest
python -m pytest
```

The test suite gradient-checks every operation against central finite
differences, so a regression in any backward rule is caught immediately.
