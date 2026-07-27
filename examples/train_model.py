import csv
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scalar_autograd import MultilayerPerceptron

DATA_PATH = Path(__file__).parent / "optdigits.csv"
MODEL_PATH = Path(__file__).parent / "optdigits_model.json"


def load_digits():
    features, labels = [], []
    with open(DATA_PATH, newline="") as f:
        for row in csv.reader(f):
            if not row:
                continue
            *pixels, label = row
            features.append([int(p) / 16.0 for p in pixels])
            labels.append(int(label))
    return features, labels


def train_test_split(n_train=500, seed=0):
    random.seed(seed)
    features, labels = load_digits()
    data = list(zip(features, labels))
    random.shuffle(data)
    features, labels = zip(*data)
    return (features[:n_train], labels[:n_train]), (features[n_train:], labels[n_train:])


def cross_entropy_loss(scores, y):
    m = max(s.data for s in scores)
    shifted = [s - m for s in scores]
    logsumexp = sum(s.exp() for s in shifted).log()
    return logsumexp - shifted[y]


def train(train_x, train_y, epochs=15):
    model = MultilayerPerceptron(64, [16, 10])
    n = len(train_x)
    weight_decay = 1e-4

    for epoch in range(epochs):
        order = list(range(n))
        random.shuffle(order)
        lr = 0.05 * (1.0 - 0.9 * epoch / epochs)

        for i in order:
            loss = cross_entropy_loss(model(train_x[i]), train_y[i])
            for p in model.parameters():
                p.grad = 0.0
            loss.backward()
            for p in model.parameters():
                p.data -= lr * (p.grad + weight_decay * p.data)

        acc = sum(model.predict(x) == y for x, y in zip(train_x, train_y)) / n
        print(f"epoch {epoch + 1:2d}/{epochs}  train accuracy {acc:.3f}")

    return model


def main():
    (train_x, train_y), _ = train_test_split()
    model = train(train_x, train_y)
    model.save(MODEL_PATH)
    print(f"\nsaved trained model to {MODEL_PATH}")


if __name__ == "__main__":
    main()
