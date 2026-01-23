
"""
Minimal ML pipeline core module
Provides a simple training/evaluation pipeline that uses PyTorch when available
and falls back to a numpy-based stub for environments without heavy ML deps.
"""

import os
import time
from typing import Any, Dict, Tuple, Optional

try:
    import torch
    from torch import nn
    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False

import numpy as np


class MLDataset:
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def get_batch(self, batch_size: int, idx: int):
        start = idx * batch_size
        end = start + batch_size
        return self.X[start:end], self.y[start:end]


class SimpleModel:
    """A tiny neural network implemented in PyTorch when available."""
    def __init__(self, input_dim: int, hidden: int = 64, output_dim: int = 1):
        self.input_dim = input_dim
        self.hidden = hidden
        self.output_dim = output_dim
        if TORCH_AVAILABLE:
            self.model = nn.Sequential(
                nn.Linear(input_dim, hidden),
                nn.ReLU(),
                nn.Linear(hidden, output_dim)
            )
        else:
            # fallback to numpy weights
            self.w1 = np.random.randn(input_dim, hidden).astype(np.float32) * 0.01
            self.b1 = np.zeros(hidden, dtype=np.float32)
            self.w2 = np.random.randn(hidden, output_dim).astype(np.float32) * 0.01
            self.b2 = np.zeros(output_dim, dtype=np.float32)

    def forward(self, x: np.ndarray):
        if TORCH_AVAILABLE:
            with torch.no_grad():
                tx = torch.from_numpy(x).float()
                out = self.model(tx).numpy()
            """
            Minimal ML pipeline core module
            Provides a simple training/evaluation pipeline that uses PyTorch when available
            and falls back to a numpy-based stub for environments without heavy ML deps.
            """

            import os
            import time
            from typing import Any, Dict, Tuple, Optional

            try:
                import torch
                from torch import nn
                TORCH_AVAILABLE = True
            except Exception:
                TORCH_AVAILABLE = False

            import numpy as np


            class MLDataset:
                def __init__(self, X: np.ndarray, y: np.ndarray):
                    self.X = X
                    self.y = y

                def __len__(self):
                    return len(self.X)

                def get_batch(self, batch_size: int, idx: int):
                    start = idx * batch_size
                    end = start + batch_size
                    return self.X[start:end], self.y[start:end]


            class SimpleModel:
                """A tiny neural network implemented in PyTorch when available."""
                def __init__(self, input_dim: int, hidden: int = 64, output_dim: int = 1):
                    self.input_dim = input_dim
                    self.hidden = hidden
                    self.output_dim = output_dim
                    if TORCH_AVAILABLE:
                        self.model = nn.Sequential(
                            nn.Linear(input_dim, hidden),
                            nn.ReLU(),
                            nn.Linear(hidden, output_dim)
                        )
                    else:
                        # fallback to numpy weights
                        self.w1 = np.random.randn(input_dim, hidden).astype(np.float32) * 0.01
                        self.b1 = np.zeros(hidden, dtype=np.float32)
                        self.w2 = np.random.randn(hidden, output_dim).astype(np.float32) * 0.01
                        self.b2 = np.zeros(output_dim, dtype=np.float32)

                def forward(self, x: np.ndarray):
                    if TORCH_AVAILABLE:
                        with torch.no_grad():
                            tx = torch.from_numpy(x).float()
                            out = self.model(tx).numpy()
                        return out
                    else:
                        h = np.maximum(0, x.dot(self.w1) + self.b1)
                        out = h.dot(self.w2) + self.b2
                        return out

                def save(self, path: str):
                    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
                    if TORCH_AVAILABLE:
                        torch.save(self.model.state_dict(), path)
                    else:
                        np.savez(path, w1=self.w1, b1=self.b1, w2=self.w2, b2=self.b2)

                def load(self, path: str):
                    if not os.path.exists(path):
                        raise FileNotFoundError(path)
                    if TORCH_AVAILABLE:
                        state = torch.load(path)
                        self.model.load_state_dict(state)
                    else:
                        data = np.load(path)
                        self.w1 = data['w1']
                        self.b1 = data['b1']
                        self.w2 = data['w2']
                        self.b2 = data['b2']


            class MLPipeline:
                def __init__(self):
                    self.model = None
                    self.history = {'loss': []}

                def prepare(self, input_dim: int, hidden: int = 64, output_dim: int = 1):
                    self.model = SimpleModel(input_dim, hidden, output_dim)

                def train(self, dataset: MLDataset, epochs: int = 10, batch_size: int = 32, lr: float = 1e-3):
                    if self.model is None:
                        raise RuntimeError('Model not prepared')

                    if TORCH_AVAILABLE:
                        # convert to torch dataset
                        X = torch.from_numpy(dataset.X).float()
                        y = torch.from_numpy(dataset.y).float()
                        dataset_t = torch.utils.data.TensorDataset(X, y)
                        loader = torch.utils.data.DataLoader(dataset_t, batch_size=batch_size, shuffle=True)
                        opt = torch.optim.Adam(self.model.model.parameters(), lr=lr)
                        loss_fn = nn.MSELoss()
                        for ep in range(epochs):
                            epoch_loss = 0.0
                            for xb, yb in loader:
                                opt.zero_grad()
                                preds = self.model.model(xb)
                                loss = loss_fn(preds.squeeze(-1), yb.float())
                                loss.backward()
                                opt.step()
                                epoch_loss += float(loss)
                            self.history['loss'].append(epoch_loss)
                    else:
                        # very simple numpy SGD for demonstration
                        X = dataset.X
                        y = dataset.y
                        n = len(X)
                        for ep in range(epochs):
                            perm = np.random.permutation(n)
                            epoch_loss = 0.0
                            for i in range(0, n, batch_size):
                                idx = perm[i:i+batch_size]
                                xb = X[idx]
                                yb = y[idx]
                                preds = self.model.forward(xb)
                                err = preds.squeeze(-1) - yb
                                loss = (err ** 2).mean()
                                epoch_loss += float(loss)
                                # rudimentary gradient step on w2 only (toy example)
                                # not intended for real training
                                grad = (xb.T @ err[:, None]) / max(1, len(xb))
                                self.model.w1 -= lr * 0.001 * grad
                            self.history['loss'].append(epoch_loss)

                def evaluate(self, dataset: MLDataset) -> Dict[str, float]:
                    X = dataset.X
                    y = dataset.y
                    preds = self.model.forward(X).squeeze(-1)
                    mse = float(((preds - y) ** 2).mean())
                    return {'mse': mse}

                def save(self, path: str):
                    if self.model is None:
                        raise RuntimeError('Model not prepared')
                    self.model.save(path)

                def load(self, path: str):
                    if self.model is None:
                        raise RuntimeError('Model not prepared')
                    self.model.load(path)


            if __name__ == '__main__':
                # small smoke test
                X = np.random.randn(100, 4).astype(np.float32)
                y = (X.sum(axis=1) + np.random.randn(100) * 0.1).astype(np.float32)
                ds = MLDataset(X, y)
                pipeline = MLPipeline()
                pipeline.prepare(input_dim=4, hidden=32, output_dim=1)
                pipeline.train(ds, epochs=2, batch_size=16)
                print('Eval:', pipeline.evaluate(ds))
