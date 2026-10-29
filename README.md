# Neural Networks from Scratch: Space Warping, Backpropagation, Adam, & Raw PyTorch Kernels

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](#)
[![NumPy](https://img.shields.io/badge/NumPy-Vectorized-green.svg)](#)
[![PyTorch Parity](https://img.shields.io/badge/Raw%20PyTorch%20Kernels-100%25%20Bitwise%20Parity-brightgreen.svg)](#)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](#)

A foundational, first-principles mathematical and practical implementation of **Deep Multilayer Perceptrons (MLPs)**, **Batch Normalization**, and **Advanced Optimizers (SGD, Momentum, RMSProp, Adam)** in both pure vectorized NumPy and **Raw PyTorch Custom `torch.autograd.Function` Kernels** (strictly zero `torch.nn` layers or `torch.optim` black boxes).

---

## Table of Contents
1. [Geometric Intuition & Mental Models](#1-geometric-intuition--mental-models)
2. [Mathematical Formulations & Rigorous Derivations](#2-mathematical-formulations--rigorous-derivations)
   - [Universal Approximation & Manifold Untangling (Space Warping)](#universal-approximation--manifold-untangling-space-warping)
   - [The 4-Equation Vectorized Backpropagation Recurrence](#the-4-equation-vectorized-backpropagation-recurrence)
   - [Weight Initialization Variance Proofs: Xavier/Glorot vs. He/Kaiming](#weight-initialization-variance-proofs-xavierglorot-vs-hekaiming)
   - [Batch Normalization Forward & Backward Gradients](#batch-normalization-forward--backward-gradients)
   - [Advanced Optimizers & The Analytical Proof of Adam Bias Correction](#advanced-optimizers--the-analytical-proof-of-adam-bias-correction)
   - [Raw PyTorch Custom `torch.autograd.Function` Kernel Architecture](#raw-pytorch-custom-torchautogradfunction-kernel-architecture)
3. [Visual Diagnostics & Empirical Verification](#3-visual-diagnostics--empirical-verification)
4. [Repository Architecture](#4-repository-architecture)
5. [Quickstart & Verification Suite](#5-quickstart--verification-suite)

---

## 1. Geometric Intuition & Mental Models

### The Origami Master: Space Warping & Manifold Untangling
A single linear model (like Logistic Regression) is limited to drawing a flat hyperplane through feature space. If the data resembles two intertwined spirals or concentric rings, no flat cut can separate the classes.
A deep neural network solves this by **physically warping, folding, stretching, and creasing space**:
1. **Affine Transformation ($`\mathbf{Z} = \mathbf{X}\mathbf{W} + \mathbf{b}`$):** Linearly stretches, rotates, shears, and shifts the coordinate grid.
2. **Non-Linear Activation ($`\mathbf{A} = \sigma(\mathbf{Z})`$ or $`\mathrm{ReLU}(\mathbf{Z})`$):** Folds space along an axis, squeezing negative coordinates to zero or bending the coordinate sheet.
3. Layer by layer, the network acts like an origami master: it untangles the intertwined spirals in higher dimensions, flattening them onto a clean subspace where a simple linear readout layer can slice cleanly between them!

### The Whisper Game: Deep Signal Propagation & Initialization
Imagine a line of 10 people whispering a message down a chain (a 10-layer neural network).
- If each person whispers slightly quieter than the last ($`\sigma = 0.01`$ small weights), the message fades into dead silence by Person 3 (**Vanishing Signal / Gradient**).
- If each person shouts louder than the last ($`\sigma = 1.0`$ large weights), by Person 3 everyone is screaming at the top of their lungs, slamming every neuron against its saturation walls $`\tanh(z) = \pm 1`$ (**Saturating Gradients**).
- **Proper Initialization (Xavier / He)** calibrates the variance of the weights such that the signal volume remains exactly constant across all 10 layers, preserving forward signal propagation and backward gradient flow!

### Navigating the Ravine: Advanced Optimizers
Loss landscapes in deep learning are plagued by pathological curvatures: steep canyon walls along one dimension and shallow flat plateaus along another.
- **Vanilla SGD:** Bounces wildly back and forth between the canyon walls while making frustratingly slow progress along the floor.
- **Polyak Momentum:** Gives the parameter update physical inertia like a heavy rolling bowling ball, averaging out the cross-canyon bounces and accelerating down the ravine.
- **RMSProp:** Dynamically divides the gradient by the running root-mean-square of recent gradients, dampening oscillations along steep axes while amplifying updates along gentle slopes.
- **Adam:** Combines momentum and adaptive per-coordinate scaling with exact mathematical **bias corrections**, allowing fast escape from flat saddle points.

---

## 2. Mathematical Formulations & Rigorous Derivations

### Universal Approximation & Manifold Untangling (Space Warping)

Let an $`L`$-layer Multilayer Perceptron have layer dimensions $`[d_0, d_1, \dots, d_L]`$ where $`\mathbf{X} = \mathbf{A}^{[0]} \in \mathbb{R}^{B \times d_0}`$.
For each layer $`l \in \{1, \dots, L\}`$:

```math
\mathbf{Z}^{[l]} = \mathbf{A}^{[l-1]} \mathbf{W}^{[l]} + \mathbf{1}\mathbf{b}^{[l]\top} \in \mathbb{R}^{B \times d_l}
```

```math
\mathbf{A}^{[l]} = g^{[l]}(\mathbf{Z}^{[l]}) \in \mathbb{R}^{B \times d_l}
```

where $`\mathbf{W}^{[l]} \in \mathbb{R}^{d_{l-1} \times d_l}`$ and $`\mathbf{b}^{[l]} \in \mathbb{R}^{d_l \times 1}`$.

By the **Cybenko-Hornik Universal Approximation Theorem**, a single hidden layer network with non-polynomial continuous activations can approximate any continuous function on compact subsets of $`\mathbb{R}^n`$ to arbitrary precision $`\epsilon > 0`$. In deeper networks, composition of foldings achieves exponential expressive efficiency with polynomial parameters.

---

### The 4-Equation Vectorized Backpropagation Recurrence

Let $`\mathcal{L}`$ be the scalar batch loss. Define the layer pre-activation adjoint error as:

```math
\boldsymbol{\delta}^{[l]} \equiv \frac{\partial \mathcal{L}}{\partial \mathbf{Z}^{[l]}} \in \mathbb{R}^{B \times d_l}
```

#### Equation 1: Output Layer Error Adjoint
For categorical cross-entropy paired with softmax logits $`\hat{\mathbf{Y}} = \mathrm{softmax}(\mathbf{Z}^{[L]})`$:

```math
\boldsymbol{\delta}^{[L]} = \frac{1}{B} (\hat{\mathbf{Y}} - \mathbf{Y})
```

#### Equation 2: Pre-Activation Error Recurrence (Backward Propagation)
For any intermediate hidden layer $`l \in \{1, \dots, L-1\}`$:

```math
\boldsymbol{\delta}^{[l]} = \left( \boldsymbol{\delta}^{[l+1]} (\mathbf{W}^{[l+1]})^\top \right) \odot g'^{[l]}(\mathbf{Z}^{[l]})
```

where $`\odot`$ denotes the elementwise Hadamard product.

#### Equation 3: Weight Matrix Gradient

```math
\frac{\partial \mathcal{L}}{\partial \mathbf{W}^{[l]}} = (\mathbf{A}^{[l-1]})^\top \boldsymbol{\delta}^{[l]} + \lambda \mathbf{W}^{[l]}
```

#### Equation 4: Bias Vector Gradient

```math
\frac{\partial \mathcal{L}}{\partial \mathbf{b}^{[l]}} = \sum_{i=1}^B \boldsymbol{\delta}_{i, \cdot}^{[l]\top}
```

---

### Weight Initialization Variance Proofs: Xavier/Glorot vs. He/Kaiming

Consider a linear neuron $`z = \sum_{i=1}^{d_{\text{in}}} w_i x_i`$. Assume $`w_i`$ and $`x_i`$ are independent, zero-mean random variables ($`\mathbb{E}[w_i] = \mathbb{E}[x_i] = 0`$).

#### Variance of a Single Product:

```math
\mathrm{Var}(w_i x_i) = \mathbb{E}[w_i^2 x_i^2] - (\mathbb{E}[w_i x_i])^2 = \mathbb{E}[w_i^2] \mathbb{E}[x_i^2] = \mathrm{Var}(w_i) \mathrm{Var}(x_i)
```

Summing across all $`d_{\text{in}}`$ inputs:

```math
\mathrm{Var}(z) = \sum_{i=1}^{d_{\text{in}}} \mathrm{Var}(w_i x_i) = d_{\text{in}} \mathrm{Var}(w) \mathrm{Var}(x)
```

#### 1. Xavier / Glorot Initialization (for Tanh / Linear / Sigmoid):
To prevent forward variance from exploding or vanishing, we require $`\mathrm{Var}(z) = \mathrm{Var}(x)`$:

```math
d_{\text{in}} \mathrm{Var}(w) = 1 \implies \mathrm{Var}(w) = \frac{1}{d_{\text{in}}}
```

To simultaneously prevent backward gradient variance from exploding or vanishing:

```math
d_{\text{out}} \mathrm{Var}(w) = 1 \implies \mathrm{Var}(w) = \frac{1}{d_{\text{out}}}
```

Taking the harmonic mean of both constraints yields **Xavier / Glorot Initialization**:

```math
\mathrm{Var}(w) = \frac{2}{d_{\text{in}} + d_{\text{out}}} \implies \mathbf{W} \sim \mathcal{N}\left(0, \; \frac{2}{d_{\text{in}} + d_{\text{out}}}\right)
```

#### 2. He / Kaiming Initialization (for ReLU):
For the Rectified Linear Unit $`a = \max(0, z)`$, negative inputs are zeroed out. Under symmetric zero-mean Gaussian pre-activations:

```math
\mathbb{E}[a^2] = \frac{1}{2} \mathrm{Var}(z)
```

```math
\mathrm{Var}(z^{[l]}) = \frac{1}{2} d_{\text{in}} \mathrm{Var}(w) \mathrm{Var}(z^{[l-1]})
```

To maintain $`\mathrm{Var}(z^{[l]}) = \mathrm{Var}(z^{[l-1]})`$, the variance must double to compensate for half the neurons dying:

```math
\mathrm{Var}(w) = \frac{2}{d_{\text{in}}} \implies \mathbf{W} \sim \mathcal{N}\left(0, \; \frac{2}{d_{\text{in}}}\right)
```

---

### Batch Normalization Forward & Backward Gradients

Given mini-batch activations $`\mathbf{X} \in \mathbb{R}^{B \times D}`$, Batch Normalization computes:

```math
\boldsymbol{\mu}_B = \frac{1}{B} \sum_{i=1}^B \mathbf{x}^{(i)}, \quad \boldsymbol{\sigma}_B^2 = \frac{1}{B} \sum_{i=1}^B (\mathbf{x}^{(i)} - \boldsymbol{\mu}_B)^2
```

```math
\hat{\mathbf{x}}^{(i)} = \frac{\mathbf{x}^{(i)} - \boldsymbol{\mu}_B}{\sqrt{\boldsymbol{\sigma}_B^2 + \epsilon}}, \quad \mathbf{y}^{(i)} = \boldsymbol{\gamma} \odot \hat{\mathbf{x}}^{(i)} + \boldsymbol{\beta}
```

#### Exact Backward Derivative:
Given upstream gradient $`\frac{\partial \mathcal{L}}{\partial \mathbf{Y}}`$:

```math
\frac{\partial \mathcal{L}}{\partial \boldsymbol{\gamma}} = \sum_{i=1}^B \frac{\partial \mathcal{L}}{\partial \mathbf{y}^{(i)}} \odot \hat{\mathbf{x}}^{(i)}, \quad \frac{\partial \mathcal{L}}{\partial \boldsymbol{\beta}} = \sum_{i=1}^B \frac{\partial \mathcal{L}}{\partial \mathbf{y}^{(i)}}
```

```math
\frac{\partial \mathcal{L}}{\partial \mathbf{x}^{(i)}} = \frac{\boldsymbol{\gamma}}{B \sqrt{\boldsymbol{\sigma}_B^2 + \epsilon}} \left[ B \frac{\partial \mathcal{L}}{\partial \mathbf{y}^{(i)}} - \sum_{j=1}^B \frac{\partial \mathcal{L}}{\partial \mathbf{y}^{(j)}} - \hat{\mathbf{x}}^{(i)} \sum_{j=1}^B \frac{\partial \mathcal{L}}{\partial \mathbf{y}^{(j)}} \odot \hat{\mathbf{x}}^{(j)} \right]
```

---

### Advanced Optimizers & The Analytical Proof of Adam Bias Correction

Let $`g_t = \nabla_\theta \mathcal{L}(\theta_t)`$ be the gradient at step $`t`$.

#### Exponential Moving Averages:

```math
m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t
```

```math
v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2
```

initialized at $`m_0 = \mathbf{0}`$ and $`v_0 = \mathbf{0}`$.

#### Proof of Bias Correction:
Unrolling $`m_t`$ back to $`m_0 = \mathbf{0}`$:

```math
m_t = (1 - \beta_1) \sum_{i=1}^t \beta_1^{t-i} g_i
```

Taking the mathematical expectation, assuming $`g_i`$ comes from a stationary distribution with expectation $`\mathbb{E}[g]`$:

```math
\mathbb{E}[m_t] = \mathbb{E}\left[ (1 - \beta_1) \sum_{i=1}^t \beta_1^{t-i} g_i \right] = \mathbb{E}[g] (1 - \beta_1) \sum_{i=1}^t \beta_1^{t-i}
```

Using the finite geometric series formula $`\sum_{k=0}^{t-1} \beta_1^k = \frac{1 - \beta_1^t}{1 - \beta_1}`$:

```math
\mathbb{E}[m_t] = \mathbb{E}[g] (1 - \beta_1) \frac{1 - \beta_1^t}{1 - \beta_1} = \mathbb{E}[g] (1 - \beta_1^t)
```

To make $`\hat{m}_t`$ an **unbiased estimator** ($`\mathbb{E}[\hat{m}_t] = \mathbb{E}[g]`$), we must divide by $`1 - \beta_1^t`$:

```math
\hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \qquad \hat{v}_t = \frac{v_t}{1 - \beta_2^t}
```

#### Adam Parameter Update:

```math
\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{\hat{v}_t} + \epsilon} \hat{m}_t
```

---

### Raw PyTorch Custom `torch.autograd.Function` Kernel Architecture

To ensure 100% first-principles construction, we implement custom PyTorch autograd kernels using only `torch.Tensor` matrix multiplication without high-level wrappers:

```python
class LinearFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, X: torch.Tensor, W: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(X, W, b)
        return X @ W + b

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        X, W, b = ctx.saved_tensors
        dX = grad_output @ W.T
        dW = X.T @ grad_output
        db = grad_output.sum(dim=0, keepdim=True)
        return dX, dW, db
```

---

## 3. Visual Diagnostics & Empirical Verification

### Diagnostic Plot 1: Geometric Space Warping Progression
Generated by [01_neural_networks_and_pytorch_scratch.ipynb](01_neural_networks_and_pytorch_scratch.ipynb).

![Neural Space Warping Progression](assets/01_neural_space_warping_progression.png)

- **Panel 1 (Input Space $`\mathbf{X}`$):** 3-class intertwined spiral arms in 2D space. Linearly inseparable (Linear models achieve $`33\%`$ random guess accuracy).
- **Panel 2 ($`\mathbf{Z}^{[1]}`$):** Affine transformations stretch and rotate the arms.
- **Panel 3 ($`\mathbf{A}^{[1]}`$):** ReLU folds the manifold along non-linear crease boundaries.
- **Panel 4 ($`\mathbf{A}^{[L-1]}`$):** In the penultimate hidden layer bottleneck, the spiral arms are completely untangled into linearly separable clusters ($`99.3\%`$ accuracy).

---

### Diagnostic Plot 2: 10-Layer Deep Signal Propagation
Generated by [01_neural_networks_and_pytorch_scratch.ipynb](01_neural_networks_and_pytorch_scratch.ipynb).

![Weight Init Activation Distributions](assets/02_weight_init_activation_distributions.png)

- **Row 1 (Small Gaussian, $`\sigma = 0.01`$):** Activation standard deviation collapses to zero by Layer 3 ($`1.2 \times 10^{-19}`$ at Layer 10), freezing backpropagation.
- **Row 2 (Large Gaussian, $`\sigma = 1.00`$):** Variance explodes, slamming every neuron against the $`\pm 1`$ saturation walls with zero gradient flow.
- **Row 3 (Xavier / Glorot):** Preserves healthy activation distributions ($`\sigma \approx 0.65 - 0.70`$) cleanly across all 10 deep layers.

---

### Diagnostic Plot 3: Saddle-Point Escape & Raw PyTorch Custom Autograd Training
Generated by [01_neural_networks_and_pytorch_scratch.ipynb](01_neural_networks_and_pytorch_scratch.ipynb).

![Optimizer Trajectories Saddle Point](assets/03_optimizer_trajectories_saddle_point.png)

- **Panel A (Saddle Point Escape):** Visualizes optimizer trajectories on a non-convex saddle-ravine landscape. Vanilla SGD stalls indefinitely on the flat saddle axis, Momentum oscillates, and Adam rapidly tracks the true negative curvature escape vector.
- **Panel B (Raw PyTorch Custom Kernel MLP):** Compares training curves of `RawPyTorchMLP` trained with custom autograd kernels, demonstrating that Adam achieves the fastest convergence down to $`0.05`$ loss.

---

## 4. Repository Architecture

```text
neural-networks-numpy-to-raw-pytorch/
├── README.md                                          # Mathematical textbook & complete guide
├── requirements.txt                                   # numpy, matplotlib, torch
├── .gitignore                                         # Git ignore rules
├── src/
│   ├── __init__.py                                    # Package exposure
│   ├── plot_theme.py                                  # GitHub Dark Mode (#0D1117) styling engine
│   └── nn_engine.py                                   # Dual NumPy & Raw PyTorch custom kernel engine
├── 01_neural_networks_and_pytorch_scratch.ipynb       # Space warping, deep init, & custom autograd kernels
└── assets/
    ├── 01_neural_space_warping_progression.png        # 300-DPI origami manifold untangling
    ├── 02_weight_init_activation_distributions.png    # 300-DPI 10-layer vanishing vs saturating
    └── 03_optimizer_trajectories_saddle_point.png     # 300-DPI 2D saddle escape & training curves
```

---

## 5. Quickstart & Verification Suite

### Installation
Clone this repository and install dependencies:
```bash
git clone https://github.com/Necromancer0912/neural-networks-numpy-to-raw-pytorch.git
cd neural-networks-numpy-to-raw-pytorch
pip install -r requirements.txt
```

### Running the Mathematical Verification Suite
Verify gradient checks and exact NumPy $`\leftrightarrow`$ PyTorch custom kernel parity:
```bash
python -m src.nn_engine
```

Expected output:
```text
========================================================================
MODULE 08: DEEP MLP (NUMPY & RAW PYTORCH) — VERIFICATION SUITE
========================================================================
[Grad Check] NumPy MLP Layer 1 dW       | Relative Error: 2.272e-09 (< 1e-7 PASS)
[Grad Check] NumPy MLP Layer 2 dW       | Relative Error: 2.782e-09 (< 1e-7 PASS)
[Grad Check] NumPy MLP Layer 3 dW       | Relative Error: 1.328e-09 (< 1e-7 PASS)
[NumPy <-> Raw PyTorch Parity] Layer 1 | Max |dW_np - dW_pt| = 1.735e-17 | Max |db_np - db_pt| = 3.469e-17
[NumPy <-> Raw PyTorch Parity] Layer 2 | Max |dW_np - dW_pt| = 2.776e-17 | Max |db_np - db_pt| = 2.776e-17
[NumPy <-> Raw PyTorch Parity] Layer 3 | Max |dW_np - dW_pt| = 1.388e-17 | Max |db_np - db_pt| = 2.776e-17
========================================================================
```

