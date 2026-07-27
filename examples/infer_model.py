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


def main():
    if not MODEL_PATH.exists():
        raise SystemExit(
            f"no model found at {MODEL_PATH}; run `python examples/train_model.py` first"
        )

    model = MultilayerPerceptron.load(MODEL_PATH)
    print(f"loaded model from {MODEL_PATH}")

    _, (test_x, test_y) = train_test_split()

    correct = sum(model.predict(x) == y for x, y in zip(test_x, test_y))
    print(f"test accuracy: {correct / len(test_y):.3f}\n")

    for x, y in list(zip(test_x, test_y))[:10]:
        print(f"predicted {model.predict(x)}  (actual {y})")


if __name__ == "__main__":
    main()
