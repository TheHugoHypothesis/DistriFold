import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score

#Codigo usado na mateira de Machine Learning, só refatorado para ficar mais fácil de chamar na classe do worker

def inicializar_weights_he(inp, out):
    return np.random.randn(out, inp) * np.sqrt(2.0 / inp)


def inicializar_weights_xavier(inp, out):
    return np.random.randn(out, inp) * np.sqrt(2.0 / (inp + out))


class MLPClassifierSimple:
    def __init__(
        self,
        input_dim,
        num_classes,
        h1=64,
        h2=16,
        lr=0.005,
        l2=1e-4,
        epochs=200,
        batch_size=128,
        dropout_rate=0.2,
        patience=20,
        seed=42,
    ):
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.h1 = h1
        self.h2 = h2
        self.lr = lr
        self.l2 = l2
        self.epochs = epochs
        self.batch_size = batch_size
        self.dropout_rate = dropout_rate
        self.patience = patience
        self.seed = seed
        self._rng = np.random.default_rng(seed)

        self.W1 = inicializar_weights_he(self.input_dim, self.h1)
        self.b1 = np.zeros(self.h1)
        self.W2 = inicializar_weights_he(self.h1, self.h2)
        self.b2 = np.zeros(self.h2)
        self.W3 = inicializar_weights_xavier(self.h2, self.num_classes)
        self.b3 = np.zeros(self.num_classes)

    def _forward(self, X):
        z1 = X @ self.W1.T + self.b1
        a1 = np.maximum(0, z1)
        z2 = a1 @ self.W2.T + self.b2
        a2 = np.maximum(0, z2)
        logits = a2 @ self.W3.T + self.b3
        exp = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        probs = exp / (exp.sum(axis=1, keepdims=True) + 1e-8)
        return z1, a1, z2, a2, probs

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        best_acc = -np.inf
        best_weights = None
        epochs_no_improve = 0
        history = []

        for ep in range(self.epochs):
            perm = self._rng.permutation(len(X_train))
            Xs, ys = X_train[perm], y_train[perm]
            batch_losses = []

            for i in range(0, len(Xs), self.batch_size):
                Xb, yb = Xs[i : i + self.batch_size], ys[i : i + self.batch_size]

                z1 = Xb @ self.W1.T + self.b1
                a1 = np.maximum(0, z1)
                if self.dropout_rate > 0:
                    mask1 = (
                        self._rng.binomial(1, 1 - self.dropout_rate, size=a1.shape)
                        / (1 - self.dropout_rate)
                    )
                    a1 *= mask1

                z2 = a1 @ self.W2.T + self.b2
                a2 = np.maximum(0, z2)
                if self.dropout_rate > 0:
                    mask2 = (
                        self._rng.binomial(1, 1 - self.dropout_rate, size=a2.shape)
                        / (1 - self.dropout_rate)
                    )
                    a2 *= mask2

                logits = a2 @ self.W3.T + self.b3
                exp = np.exp(logits - np.max(logits, axis=1, keepdims=True))
                probs = exp / (exp.sum(axis=1, keepdims=True) + 1e-8)

                loss = -np.mean(np.log(probs[np.arange(len(yb)), yb] + 1e-8))
                total_loss = loss + 0.5 * self.l2 * (
                    np.sum(self.W1 * self.W1)
                    + np.sum(self.W2 * self.W2)
                    + np.sum(self.W3 * self.W3)
                )
                batch_losses.append(total_loss)

                g3 = probs.copy()
                g3[np.arange(len(yb)), yb] -= 1
                g3 /= len(yb)
                g2 = (g3 @ self.W3) * (z2 > 0)
                if self.dropout_rate > 0:
                    g2 *= mask2
                g1 = (g2 @ self.W2) * (z1 > 0)
                if self.dropout_rate > 0:
                    g1 *= mask1

                self.W3 -= self.lr * (g3.T @ a2 + self.l2 * self.W3)
                self.b3 -= self.lr * g3.sum(axis=0)
                self.W2 -= self.lr * (g2.T @ a1 + self.l2 * self.W2)
                self.b2 -= self.lr * g2.sum(axis=0)
                self.W1 -= self.lr * (g1.T @ Xb + self.l2 * self.W1)
                self.b1 -= self.lr * g1.sum(axis=0)

            train_loss = float(np.mean(batch_losses))

            val_metrics = None
            if X_val is not None and y_val is not None:
                val_metrics = self.evaluate(X_val, y_val)
                val_acc = val_metrics["accuracy"]
                if val_acc > best_acc:
                    best_acc = val_acc
                    best_weights = self.get_weights()
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1
            else:
                best_weights = self.get_weights()

            history.append({"epoch": ep, "train_loss": train_loss, "val": val_metrics})

            if X_val is not None and y_val is not None and epochs_no_improve >= self.patience:
                break

        if best_weights is not None:
            self.set_weights(best_weights)

        return {"history": history}

    def predict_proba(self, X):
        _, _, _, _, probs = self._forward(X)
        return probs

    def predict(self, X):
        return self.predict_proba(X).argmax(axis=1)

    def evaluate(self, X, y):
        probs = self.predict_proba(X)
        loss = -np.mean(np.log(probs[np.arange(len(y)), y] + 1e-8))
        pred = probs.argmax(axis=1)
        return {
            "accuracy": float((pred == y).mean()),
            "precision": float(precision_score(y, pred, average="weighted", zero_division=0)),
            "recall": float(recall_score(y, pred, average="weighted", zero_division=0)),
            "f1": float(f1_score(y, pred, average="weighted", zero_division=0)),
            "loss": float(loss),
        }

    def get_weights(self):
        return {
            "W1": self.W1.copy(),
            "b1": self.b1.copy(),
            "W2": self.W2.copy(),
            "b2": self.b2.copy(),
            "W3": self.W3.copy(),
            "b3": self.b3.copy(),
        }

    def set_weights(self, weights):
        self.W1 = weights["W1"].copy()
        self.b1 = weights["b1"].copy()
        self.W2 = weights["W2"].copy()
        self.b2 = weights["b2"].copy()
        self.W3 = weights["W3"].copy()
        self.b3 = weights["b3"].copy()


def train_fold_from_arrays(X, y, train_idx, test_idx, config, fold_id=None):
    X_train = X[train_idx]
    y_train = y[train_idx].astype(int)
    X_test = X[test_idx]
    y_test = y[test_idx].astype(int)

    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0) + 1e-8
    X_train = (X_train - mean) / std
    X_test = (X_test - mean) / std

    num_classes = int(np.max(y_train)) + 1
    input_dim = X_train.shape[1]

    model = MLPClassifierSimple(
        input_dim=input_dim,
        num_classes=num_classes,
        h1=config.get("h1", 64),
        h2=config.get("h2", 16),
        lr=config.get("lr", 0.005),
        l2=config.get("l2", 1e-4),
        epochs=config.get("epochs", 200),
        batch_size=config.get("batch_size", 128),
        dropout_rate=config.get("dropout_rate", 0.2),
        patience=config.get("patience", 20),
        seed=config.get("seed", 42),
    )

    model.fit(X_train, y_train, X_test, y_test)
    metrics = model.evaluate(X_test, y_test)
    result = {"metrics": metrics, "weights": model.get_weights()}
    if fold_id is not None:
        result["fold_id"] = fold_id
    return result

