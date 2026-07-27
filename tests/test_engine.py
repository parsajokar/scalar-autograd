import math

import pytest

from scalar_autograd import Value


def check_gradients(expr, inits, tol=1e-4, eps=1e-6):
    xs = [Value(v) for v in inits]
    out = expr(xs)
    out.backward()
    analytic = [x.grad for x in xs]

    for i in range(len(inits)):
        plus = list(inits)
        minus = list(inits)
        plus[i] += eps
        minus[i] -= eps
        f_plus = expr([Value(v) for v in plus]).data
        f_minus = expr([Value(v) for v in minus]).data
        numeric = (f_plus - f_minus) / (2 * eps)
        assert analytic[i] == pytest.approx(numeric, abs=tol), (
            f"grad mismatch on input {i}: analytic={analytic[i]} numeric={numeric}"
        )


def test_forward_arithmetic():
    a, b = Value(2.0), Value(-3.0)
    assert (a + b).data == -1.0
    assert (a * b).data == -6.0
    assert (a - b).data == 5.0
    assert (a ** 3).data == 8.0
    assert (b / a).data == -1.5
    assert (-a).data == -2.0


def test_forward_with_python_scalars():
    a = Value(4.0)
    assert (a + 1).data == 5.0
    assert (1 + a).data == 5.0
    assert (2 * a).data == 8.0
    assert (10 - a).data == 6.0
    assert (8 / a).data == 2.0


def test_relu_forward():
    assert Value(3.0).relu().data == 3.0
    assert Value(-3.0).relu().data == 0.0
    assert Value(0.0).relu().data == 0.0


def test_exp_log_tanh_forward():
    assert Value(0.0).exp().data == pytest.approx(1.0)
    assert Value(1.0).exp().data == pytest.approx(math.e)
    assert Value(1.0).log().data == pytest.approx(0.0)
    assert Value(math.e).log().data == pytest.approx(1.0)
    assert Value(0.0).tanh().data == pytest.approx(0.0)
    assert Value(math.log(3)).exp().data == pytest.approx(3.0)


def test_backward_sets_root_grad():
    x = Value(5.0)
    y = x * 2
    y.backward()
    assert y.grad == 1.0
    assert x.grad == 2.0


def test_grad_add_mul():
    check_gradients(lambda x: x[0] * x[1] + x[0], [2.0, -3.0])


def test_grad_pow():
    check_gradients(lambda x: x[0] ** 3 + x[1] ** 2, [1.5, -2.0])


def test_grad_div():
    check_gradients(lambda x: x[0] / x[1], [3.0, 4.0])


def test_grad_reverse_ops():
    check_gradients(lambda x: (2 - x[0]) + (10 / x[1]) + (3 * x[2]), [1.0, 5.0, -2.0])


def test_grad_exp():
    check_gradients(lambda x: (x[0] * x[1]).exp() + x[0].exp(), [0.7, -1.3])


def test_grad_log():
    check_gradients(lambda x: (x[0] * x[1]).log() + x[1].log(), [1.5, 2.0])


def test_grad_tanh():
    check_gradients(lambda x: (x[0] * x[1]).tanh() + x[0].tanh(), [0.4, -0.9])


def test_grad_softmax_cross_entropy():
    def expr(x):
        logsumexp = sum(s.exp() for s in x).log()
        return logsumexp - x[1]

    check_gradients(expr, [0.5, -1.0, 2.0])


def test_grad_relu_positive_and_negative():
    check_gradients(lambda x: (x[0] * x[1]).relu() + x[2].relu(), [2.0, 3.0, -1.5])


def test_grad_variable_reused():
    check_gradients(lambda x: x[0] * x[0] + x[0] * 3 + x[0] ** 2, [2.5])


def test_grad_small_network():
    def expr(x):
        h1 = (x[0] * 2 + x[1] * -1 + 1).relu()
        h2 = (x[0] * -3 + x[1] * 4 - 2).relu()
        return h1 * x[2] + h2 * x[3]

    check_gradients(expr, [0.5, -0.7, 1.2, -0.3])


def test_grad_matches_known_micrograd_case():
    a = Value(-4.0)
    b = Value(2.0)
    c = a + b
    d = a * b + b ** 3
    c = c + c + 1
    c = c + 1 + c + (-a)
    d = d + d * 2 + (b + a).relu()
    d = d + 3 * d + (b - a).relu()
    e = c - d
    f = e ** 2
    g = f / 2.0
    g = g + 10.0 / f
    g.backward()

    assert g.data == pytest.approx(24.7040, abs=1e-4)
    assert a.grad == pytest.approx(138.8338, abs=1e-4)
    assert b.grad == pytest.approx(645.5773, abs=1e-4)
