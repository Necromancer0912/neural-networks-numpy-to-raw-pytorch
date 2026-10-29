"""
Neural Networks From Scratch (`src/`).

Dual first-principles implementation:
1. Pure NumPy Deep Multilayer Perceptron (`NumPyDeepMLPScratch`), BatchNorm (`BatchNorm1DScratch`),
   and Optimizers (`NumPyOptimizerScratch` with SGD, Momentum, RMSProp, Adam).
2. Raw PyTorch Custom Kernels (`LinearFunction`, `ReLUFunction`, `GELUFunction`, `StableCrossEntropyFunction`)
   and Raw PyTorch Models (`RawPyTorchMLP`, `RawPyTorchOptimizer`) using only `torch.Tensor` operations.
"""

from .nn_engine import (
    BatchNorm1DScratch,
    GELUFunction,
    LinearFunction,
    NumPyDeepMLPScratch,
    NumPyOptimizerScratch,
    RawPyTorchMLP,
    RawPyTorchOptimizer,
    StableCrossEntropyFunction,
    np_activation_backward,
    np_activation_forward,
)
from .plot_theme import (
    AMBER,
    BG_AXES,
    BG_CANVAS,
    CLASS_COLORS,
    CMAP_DECISION,
    CMAP_MULTICLASS,
    CMAP_SURFACE,
    CORAL,
    CYAN,
    EMERALD,
    MAGENTA,
    PALETTE,
    PURPLE,
    SLATE_BLUE,
    TEXT_MUTED,
    TEXT_PRIMARY,
    apply_github_dark_style,
    plot_decision_boundary,
    plot_loss_landscape,
    save_fig,
    style_3d_axis,
)

__all__ = [
    "np_activation_forward",
    "np_activation_backward",
    "BatchNorm1DScratch",
    "NumPyDeepMLPScratch",
    "NumPyOptimizerScratch",
    "LinearFunction",
    "GELUFunction",
    "StableCrossEntropyFunction",
    "RawPyTorchMLP",
    "RawPyTorchOptimizer",
    "apply_github_dark_style",
    "style_3d_axis",
    "plot_decision_boundary",
    "plot_loss_landscape",
    "save_fig",
    "PALETTE",
    "CLASS_COLORS",
    "BG_CANVAS",
    "BG_AXES",
    "TEXT_PRIMARY",
    "TEXT_MUTED",
    "CYAN",
    "CORAL",
    "AMBER",
    "EMERALD",
    "PURPLE",
    "SLATE_BLUE",
    "MAGENTA",
    "CMAP_SURFACE",
    "CMAP_DECISION",
    "CMAP_MULTICLASS",
]
