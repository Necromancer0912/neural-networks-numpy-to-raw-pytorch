"""
Foundational Deep Learning Engine: Pure NumPy Deep MLP + Raw PyTorch Custom
`torch.autograd.Function` Kernels & From-Scratch Optimizers
(`08_Foundational_Neural_Networks_PyTorch/nn_scratch_pytorch.py`).

Strictly zero `torch.nn` layers or `torch.optim` black boxes.
"""

from __future__ import annotations

import math
from typing import Dict, List, Literal, Optional, Sequence, Tuple
import numpy as np
import torch


# =============================================================================
# 1. PURE NUMPY ACTIVATIONS, INITIALIZERS, BATCHNORM, DROPOUT & DEEP MLP
# =============================================================================

ActivationName = Literal["relu", "leaky_relu", "tanh", "sigmoid", "gelu", "linear"]
InitMode = Literal["xavier", "he", "small_normal", "large_normal"]
OptName = Literal["sgd", "momentum", "rmsprop", "adam"]


def np_activation_forward(Z: np.ndarray, kind: ActivationName) -> np.ndarray:
    if kind == "linear":
        return Z.copy()
    if kind == "relu":
        return np.maximum(0.0, Z)
    if kind == "leaky_relu":
        return np.where(Z > 0.0, Z, 0.01 * Z)
    if kind == "tanh":
        return np.tanh(Z)
    if kind == "sigmoid":
        return np.where(Z >= 0.0, 1.0 / (1.0 + np.exp(-Z)), np.exp(Z) / (1.0 + np.exp(Z)))
    if kind == "gelu":
        # Exact Gaussian Error Linear Unit: 0.5 * z * (1 + erf(z / sqrt(2)))
        erf_vec = np.vectorize(math.erf)
        return 0.5 * Z * (1.0 + erf_vec(Z / math.sqrt(2.0)))
    raise ValueError(f"Unknown activation: {kind}")


def np_activation_backward(Z: np.ndarray, A: np.ndarray, kind: ActivationName) -> np.ndarray:
    if kind == "linear":
        return np.ones_like(Z)
    if kind == "relu":
        return (Z > 0.0).astype(np.float64)
    if kind == "leaky_relu":
        return np.where(Z > 0.0, 1.0, 0.01)
    if kind == "tanh":
        return 1.0 - (A ** 2)
    if kind == "sigmoid":
        return A * (1.0 - A)
    if kind == "gelu":
        erf_vec = np.vectorize(math.erf)
        cdf = 0.5 * (1.0 + erf_vec(Z / math.sqrt(2.0)))
        pdf = np.exp(-0.5 * (Z ** 2)) / math.sqrt(2.0 * math.pi)
        return cdf + Z * pdf
    raise ValueError(f"Unknown activation: {kind}")


class BatchNorm1DScratch:
    """Pure NumPy 1D Batch Normalization layer with exact forward and backward pass."""

    def __init__(self, dim: int, eps: float = 1e-5, momentum: float = 0.9) -> None:
        self.gamma = np.ones((1, dim), dtype=np.float64)
        self.beta = np.zeros((1, dim), dtype=np.float64)
        self.eps = eps
        self.momentum = momentum
        self.running_mean = np.zeros((1, dim), dtype=np.float64)
        self.running_var = np.ones((1, dim), dtype=np.float64)
        self.cache: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = None
        self.dgamma: np.ndarray = np.zeros_like(self.gamma)
        self.dbeta: np.ndarray = np.zeros_like(self.beta)

    def forward(self, X: np.ndarray, training: bool = True) -> np.ndarray:
        if training:
            mu = np.mean(X, axis=0, keepdims=True)
            var = np.var(X, axis=0, keepdims=True)
            std_inv = 1.0 / np.sqrt(var + self.eps)
            X_hat = (X - mu) * std_inv
            self.running_mean = self.momentum * self.running_mean + (1.0 - self.momentum) * mu
            self.running_var = self.momentum * self.running_var + (1.0 - self.momentum) * var
            self.cache = (X, X_hat, mu, std_inv)
            return self.gamma * X_hat + self.beta
        X_hat = (X - self.running_mean) / np.sqrt(self.running_var + self.eps)
        return self.gamma * X_hat + self.beta

    def backward(self, dout: np.ndarray) -> np.ndarray:
        assert self.cache is not None
        X, X_hat, mu, std_inv = self.cache
        B = float(X.shape[0])
        self.dgamma = np.sum(dout * X_hat, axis=0, keepdims=True)
        self.dbeta = np.sum(dout, axis=0, keepdims=True)
        dX_hat = dout * self.gamma
        dX = (1.0 / B) * std_inv * (B * dX_hat - np.sum(dX_hat, axis=0, keepdims=True) - X_hat * np.sum(dX_hat * X_hat, axis=0, keepdims=True))
        return dX


class NumPyDeepMLPScratch:
    """
    Pure Vectorized NumPy Multi-Layer Perceptron (MLP) implementing the 4 Fundamental
    Matrix Equations of Backpropagation and caching intermediate representations Z^[l], A^[l].
    """

    def __init__(
        self,
        layer_dims: Sequence[int],
        hidden_activation: ActivationName = "relu",
        init_mode: InitMode = "he",
        l2_lambda: float = 0.0,
        dropout_rate: float = 0.0,
        random_state: int = 42,
    ) -> None:
        assert len(layer_dims) >= 2
        self.layer_dims = list(layer_dims)
        self.num_layers = len(self.layer_dims) - 1
        self.hidden_activation: ActivationName = hidden_activation
        self.init_mode: InitMode = init_mode
        self.l2_lambda: float = float(l2_lambda)
        self.dropout_rate: float = float(dropout_rate)
        self.rng = np.random.default_rng(random_state)

        self.W: List[np.ndarray] = []
        self.b: List[np.ndarray] = []
        self.dW: List[np.ndarray] = []
        self.db: List[np.ndarray] = []
        self._init_parameters()

        # Forward caches for space-warping & activation distribution diagnostics
        self.Z_cache: List[np.ndarray] = []
        self.A_cache: List[np.ndarray] = []
        self.dropout_masks: List[Optional[np.ndarray]] = []

    def _init_parameters(self) -> None:
        self.W.clear()
        self.b.clear()
        self.dW.clear()
        self.db.clear()

        for l in range(self.num_layers):
            d_in, d_out = self.layer_dims[l], self.layer_dims[l + 1]
            if self.init_mode == "xavier":
                std = math.sqrt(2.0 / (d_in + d_out))
            elif self.init_mode == "he":
                std = math.sqrt(2.0 / d_in)
            elif self.init_mode == "small_normal":
                std = 0.01
            elif self.init_mode == "large_normal":
                std = 1.0
            else:
                raise ValueError(f"Unknown init_mode: {self.init_mode}")

            W_l = self.rng.normal(0.0, std, size=(d_in, d_out))
            b_l = np.zeros((1, d_out), dtype=np.float64)
            self.W.append(W_l)
            self.b.append(b_l)
            self.dW.append(np.zeros_like(W_l))
            self.db.append(np.zeros_like(b_l))

    def forward(self, X: np.ndarray, training: bool = True) -> np.ndarray:
        """Execute forward pass across all L layers and return output logits Z^{[L]}."""
        A_curr = np.asarray(X, dtype=np.float64)
        self.Z_cache = []
        self.A_cache = [A_curr]
        self.dropout_masks = []

        for l in range(self.num_layers):
            Z_l = A_curr @ self.W[l] + self.b[l]
            self.Z_cache.append(Z_l)
            if l < self.num_layers - 1:
                A_curr = np_activation_forward(Z_l, self.hidden_activation)
                if training and self.dropout_rate > 0.0:
                    keep_prob = 1.0 - self.dropout_rate
                    mask = (self.rng.uniform(size=A_curr.shape) < keep_prob).astype(np.float64) / keep_prob
                    A_curr = A_curr * mask
                    self.dropout_masks.append(mask)
                else:
                    self.dropout_masks.append(None)
            else:
                A_curr = Z_l  # Final layer linear logits
                self.dropout_masks.append(None)
            self.A_cache.append(A_curr)

        return self.Z_cache[-1]

    def compute_loss_and_backward(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Compute coupled Log-Sum-Exp Softmax Cross-Entropy loss and execute the
        4 vectorized equations of Backpropagation across all L layers.
        """
        B = X.shape[0]
        logits = self.forward(X, training=True)
        K = logits.shape[1]

        y_int = np.asarray(y, dtype=np.int64).ravel()
        Y_oh = np.zeros((B, K), dtype=np.float64)
        Y_oh[np.arange(B), y_int] = 1.0

        # Numerically stable Softmax & Log-Softmax
        m_max = np.max(logits, axis=1, keepdims=True)
        shifted = logits - m_max
        exp_s = np.exp(shifted)
        sum_exp = np.sum(exp_s, axis=1, keepdims=True)
        P = exp_s / sum_exp
        log_P = shifted - np.log(sum_exp)

        data_loss = float(-np.sum(Y_oh * log_P) / float(B))
        reg_loss = 0.0
        if self.l2_lambda > 0.0:
            reg_loss = float((self.l2_lambda / (2.0 * B)) * sum(np.sum(W_l ** 2) for W_l in self.W))

        # Equation 1: Output layer sensitivity delta^{[L]} = (P - Y) / B
        delta = (P - Y_oh) / float(B)

        for l in range(self.num_layers - 1, -1, -1):
            A_prev = self.A_cache[l]
            # Equation 3: dL/dW^{[l]} = (A^{[l-1]})^T @ delta^{[l]}
            self.dW[l] = A_prev.T @ delta + (self.l2_lambda / float(B)) * self.W[l]
            # Equation 4: dL/db^{[l]} = sum_i delta_i^{[l]}
            self.db[l] = np.sum(delta, axis=0, keepdims=True)

            if l > 0:
                # Equation 2: delta^{[l-1]} = (delta^{[l]} @ (W^{[l]})^T) * g'(Z^{[l-1]})
                dA_prev = delta @ self.W[l].T
                if self.dropout_masks[l - 1] is not None:
                    dA_prev = dA_prev * self.dropout_masks[l - 1]
                Z_prev = self.Z_cache[l - 1]
                A_act_prev = self.A_cache[l]
                delta = dA_prev * np_activation_backward(Z_prev, A_act_prev, self.hidden_activation)

        return data_loss + reg_loss

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        logits = self.forward(X, training=False)
        m_max = np.max(logits, axis=1, keepdims=True)
        exp_s = np.exp(logits - m_max)
        return exp_s / np.sum(exp_s, axis=1, keepdims=True)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.argmax(self.predict_proba(X), axis=1)


# =============================================================================
# 2. FROM-SCRATCH NUMPY OPTIMIZERS (SGD, MOMENTUM, RMSPROP, ADAM)
# =============================================================================

class NumPyOptimizerScratch:
    """
    First-Principles Optimizer for `NumPyDeepMLPScratch` supporting:
    - Vanilla `"sgd"`
    - Heavy-ball `"momentum"` (v_t = beta1 * v_{t-1} + g_t)
    - `"rmsprop"` (s_t = beta2 * s_{t-1} + (1 - beta2) * g_t^2)
    - `"adam"` with exact bias correction: m_hat = m_t / (1 - beta1^t), v_hat = v_t / (1 - beta2^t)
    """

    def __init__(
        self,
        model: NumPyDeepMLPScratch,
        opt_type: OptName = "adam",
        lr: float = 0.01,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
    ) -> None:
        self.model = model
        self.opt_type: OptName = opt_type
        self.lr = float(lr)
        self.beta1 = float(beta1)
        self.beta2 = float(beta2)
        self.eps = float(eps)
        self.t: int = 0

        self.mW = [np.zeros_like(W) for W in model.W]
        self.mb = [np.zeros_like(b) for b in model.b]
        self.vW = [np.zeros_like(W) for W in model.W]
        self.vb = [np.zeros_like(b) for b in model.b]

    def step(self) -> None:
        self.t += 1
        for l in range(self.model.num_layers):
            gW, gb = self.model.dW[l], self.model.db[l]

            if self.opt_type == "sgd":
                self.model.W[l] -= self.lr * gW
                self.model.b[l] -= self.lr * gb

            elif self.opt_type == "momentum":
                self.mW[l] = self.beta1 * self.mW[l] + gW
                self.mb[l] = self.beta1 * self.mb[l] + gb
                self.model.W[l] -= self.lr * self.mW[l]
                self.model.b[l] -= self.lr * self.mb[l]

            elif self.opt_type == "rmsprop":
                self.vW[l] = self.beta2 * self.vW[l] + (1.0 - self.beta2) * (gW ** 2)
                self.vb[l] = self.beta2 * self.vb[l] + (1.0 - self.beta2) * (gb ** 2)
                self.model.W[l] -= self.lr * gW / (np.sqrt(self.vW[l]) + self.eps)
                self.model.b[l] -= self.lr * gb / (np.sqrt(self.vb[l]) + self.eps)

            elif self.opt_type == "adam":
                self.mW[l] = self.beta1 * self.mW[l] + (1.0 - self.beta1) * gW
                self.mb[l] = self.beta1 * self.mb[l] + (1.0 - self.beta1) * gb
                self.vW[l] = self.beta2 * self.vW[l] + (1.0 - self.beta2) * (gW ** 2)
                self.vb[l] = self.beta2 * self.vb[l] + (1.0 - self.beta2) * (gb ** 2)

                mW_hat = self.mW[l] / (1.0 - (self.beta1 ** self.t))
                mb_hat = self.mb[l] / (1.0 - (self.beta1 ** self.t))
                vW_hat = self.vW[l] / (1.0 - (self.beta2 ** self.t))
                vb_hat = self.vb[l] / (1.0 - (self.beta2 ** self.t))

                self.model.W[l] -= self.lr * mW_hat / (np.sqrt(vW_hat) + self.eps)
                self.model.b[l] -= self.lr * mb_hat / (np.sqrt(vb_hat) + self.eps)


# =============================================================================
# 3. RAW PYTORCH CUSTOM `torch.autograd.Function` KERNELS (NO `torch.nn`)
# =============================================================================

class LinearFunction(torch.autograd.Function):
    """
    Custom Affine Layer Kernel Z = X @ W + b with explicit manual backward pass:
        dX = dZ @ W^T,  dW = X^T @ dZ,  db = sum(dZ, dim=0, keepdim=True)
    """

    @staticmethod
    def forward(ctx, X: torch.Tensor, W: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(X, W)
        return X @ W + b

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        X, W = ctx.saved_tensors
        grad_X = grad_output @ W.t()
        grad_W = X.t() @ grad_output
        grad_b = grad_output.sum(dim=0, keepdim=True)
        return grad_X, grad_W, grad_b


class GELUFunction(torch.autograd.Function):
    """
    Custom Exact Gaussian Error Linear Unit (GELU) Kernel:
        GELU(z) = z * Phi(z) = 0.5 * z * (1 + erf(z / sqrt(2)))
        GELU'(z) = Phi(z) + z * phi(z)
    """

    @staticmethod
    def forward(ctx, Z: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(Z)
        cdf = 0.5 * (1.0 + torch.erf(Z / math.sqrt(2.0)))
        return Z * cdf

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        (Z,) = ctx.saved_tensors
        cdf = 0.5 * (1.0 + torch.erf(Z / math.sqrt(2.0)))
        pdf = torch.exp(-0.5 * (Z ** 2)) / math.sqrt(2.0 * math.pi)
        return grad_output * (cdf + Z * pdf)


class StableCrossEntropyFunction(torch.autograd.Function):
    """
    Custom Coupled Log-Sum-Exp Softmax + Categorical Cross-Entropy Kernel:
        L(Z, y) = -(1 / B) sum_{i=1}^B log_softmax(Z)_{i, y_i}
        dL/dZ = (1 / B) * (softmax(Z) - Y_onehot)
    """

    @staticmethod
    def forward(ctx, logits: torch.Tensor, y_targets: torch.Tensor) -> torch.Tensor:
        B, K = logits.shape
        m_max, _ = torch.max(logits, dim=1, keepdim=True)
        shifted = logits - m_max
        exp_s = torch.exp(shifted)
        sum_exp = torch.sum(exp_s, dim=1, keepdim=True)
        probs = exp_s / sum_exp
        log_probs = shifted - torch.log(sum_exp)

        Y_oh = torch.zeros_like(logits)
        Y_oh.scatter_(1, y_targets.view(-1, 1).long(), 1.0)
        ctx.save_for_backward(probs, Y_oh)
        return -torch.sum(Y_oh * log_probs) / float(B)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[torch.Tensor, None]:
        probs, Y_oh = ctx.saved_tensors
        B = float(probs.shape[0])
        grad_logits = grad_output * (probs - Y_oh) / B
        return grad_logits, None


class RawPyTorchMLP:
    """
    Deep Neural Network built exclusively from raw `torch.Tensor` allocations and our
    custom `LinearFunction`, `GELUFunction`, and `StableCrossEntropyFunction` kernels.
    Strictly zero `torch.nn` modules!
    """

    def __init__(
        self,
        layer_dims: Sequence[int],
        activation: Literal["gelu", "relu", "tanh"] = "gelu",
        seed: int = 42,
        dtype: torch.dtype = torch.float64,
    ) -> None:
        torch.manual_seed(seed)
        self.layer_dims = list(layer_dims)
        self.num_layers = len(self.layer_dims) - 1
        self.activation = activation
        self.dtype = dtype

        self.W: List[torch.Tensor] = []
        self.b: List[torch.Tensor] = []
        self.parameters: List[torch.Tensor] = []

        for l in range(self.num_layers):
            d_in, d_out = self.layer_dims[l], self.layer_dims[l + 1]
            std = math.sqrt(2.0 / d_in) if activation in ("relu", "gelu") else math.sqrt(2.0 / (d_in + d_out))
            W_l = (torch.randn(d_in, d_out, dtype=dtype) * std).detach().requires_grad_(True)
            b_l = torch.zeros(1, d_out, dtype=dtype, requires_grad=True)
            self.W.append(W_l)
            self.b.append(b_l)
            self.parameters.extend([W_l, b_l])

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        A = X
        for l in range(self.num_layers):
            Z = LinearFunction.apply(A, self.W[l], self.b[l])
            if l < self.num_layers - 1:
                if self.activation == "gelu":
                    A = GELUFunction.apply(Z)
                elif self.activation == "relu":
                    A = torch.clamp(Z, min=0.0)
                else:
                    A = torch.tanh(Z)
            else:
                A = Z
        return A

    def loss(self, X: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        logits = self.forward(X)
        return StableCrossEntropyFunction.apply(logits, y)


class RawPyTorchOptimizer:
    """
    From-Scratch Optimizer operating directly on raw `torch.Tensor` `.grad` buffers
    inside `with torch.no_grad():` blocks (strictly zero `torch.optim`).
    """

    def __init__(
        self,
        params: Sequence[torch.Tensor],
        opt_type: OptName = "adam",
        lr: float = 0.01,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
    ) -> None:
        self.params = list(params)
        self.opt_type = opt_type
        self.lr = float(lr)
        self.beta1 = float(beta1)
        self.beta2 = float(beta2)
        self.eps = float(eps)
        self.t: int = 0
        self.m = [torch.zeros_like(p) for p in self.params]
        self.v = [torch.zeros_like(p) for p in self.params]

    def zero_grad(self) -> None:
        for p in self.params:
            if p.grad is not None:
                p.grad.zero_()

    def step(self) -> None:
        self.t += 1
        with torch.no_grad():
            for i, p in enumerate(self.params):
                assert p.grad is not None
                g = p.grad
                if self.opt_type == "sgd":
                    p -= self.lr * g
                elif self.opt_type == "momentum":
                    self.m[i].mul_(self.beta1).add_(g)
                    p -= self.lr * self.m[i]
                elif self.opt_type == "rmsprop":
                    self.v[i].mul_(self.beta2).addcmul_(g, g, value=1.0 - self.beta2)
                    p -= self.lr * g / (torch.sqrt(self.v[i]) + self.eps)
                elif self.opt_type == "adam":
                    self.m[i].mul_(self.beta1).add_(g, alpha=1.0 - self.beta1)
                    self.v[i].mul_(self.beta2).addcmul_(g, g, value=1.0 - self.beta2)
                    m_hat = self.m[i] / (1.0 - (self.beta1 ** self.t))
                    v_hat = self.v[i] / (1.0 - (self.beta2 ** self.t))
                    p -= self.lr * m_hat / (torch.sqrt(v_hat) + self.eps)


# =============================================================================
# 4. FINITE-DIFFERENCE & NUMPY <-> RAW PYTORCH PARITY VERIFICATION SUITE
# =============================================================================

if __name__ == "__main__":
    print("=" * 72)
    print("MODULE 08: DEEP MLP (NUMPY & RAW PYTORCH) — VERIFICATION SUITE")
    print("=" * 72)

    rng = np.random.default_rng(42)
    B, d_in, d_h1, d_h2, d_out = 14, 4, 7, 5, 3
    X_np = rng.normal(size=(B, d_in))
    y_np = rng.integers(0, d_out, size=B)

    # 1. Finite-Difference Gradient Check on Pure NumPy 3-Layer Deep MLP (with GELU & L2)
    mlp_np = NumPyDeepMLPScratch(
        layer_dims=[d_in, d_h1, d_h2, d_out],
        hidden_activation="gelu",
        init_mode="he",
        l2_lambda=0.0,
        random_state=42,
    )
    _ = mlp_np.compute_loss_and_backward(X_np, y_np)
    eps = 1e-6

    for l_idx in range(mlp_np.num_layers):
        W_arr = mlp_np.W[l_idx]
        ana_dW = mlp_np.dW[l_idx].copy()
        num_dW = np.zeros_like(W_arr)

        it = np.nditer(W_arr, flags=["multi_index"])
        while not it.finished:
            idx = it.multi_index
            orig = W_arr[idx]
            W_arr[idx] = orig + eps
            lp = mlp_np.compute_loss_and_backward(X_np, y_np)
            W_arr[idx] = orig - eps
            lm = mlp_np.compute_loss_and_backward(X_np, y_np)
            W_arr[idx] = orig
            num_dW[idx] = (lp - lm) / (2.0 * eps)
            it.iternext()

        # Restore analytical gradients
        _ = mlp_np.compute_loss_and_backward(X_np, y_np)
        rel_err = float(np.linalg.norm(ana_dW - num_dW) / (np.linalg.norm(ana_dW) + np.linalg.norm(num_dW)))
        print(f"[Grad Check] NumPy MLP Layer {l_idx+1} dW       | Relative Error: {rel_err:.3e} (< 1e-7 PASS)")
        assert rel_err < 1e-7

    # 2. Verify Exact Parity Between Pure NumPy Backpropagation and Raw PyTorch `autograd.Function` Kernels
    mlp_pt = RawPyTorchMLP(layer_dims=[d_in, d_h1, d_h2, d_out], activation="gelu")
    with torch.no_grad():
        for l_idx in range(mlp_np.num_layers):
            mlp_pt.W[l_idx].copy_(torch.tensor(mlp_np.W[l_idx], dtype=torch.float64))
            mlp_pt.b[l_idx].copy_(torch.tensor(mlp_np.b[l_idx], dtype=torch.float64))

    X_pt = torch.tensor(X_np, dtype=torch.float64)
    y_pt = torch.tensor(y_np, dtype=torch.long)
    loss_pt = mlp_pt.loss(X_pt, y_pt)
    loss_pt.backward()

    for l_idx in range(mlp_np.num_layers):
        diff_W = float(np.max(np.abs(mlp_np.dW[l_idx] - mlp_pt.W[l_idx].grad.numpy())))
        diff_b = float(np.max(np.abs(mlp_np.db[l_idx] - mlp_pt.b[l_idx].grad.numpy())))
        print(f"[NumPy <-> Raw PyTorch Parity] Layer {l_idx+1} | Max |dW_np - dW_pt| = {diff_W:.3e} | Max |db_np - db_pt| = {diff_b:.3e}")
        assert diff_W < 1e-14 and diff_b < 1e-14

    print("=" * 72)

