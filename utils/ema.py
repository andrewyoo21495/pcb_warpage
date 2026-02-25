#!/usr/bin/env python3
"""Exponential Moving Average (EMA) for model parameters.

Essential for DDPM generation quality.

Usage during training:
    ema = EMA(model, decay=0.9999)
    for batch in loader:
        optimizer.step()
        ema.update()

    # Save checkpoint with EMA weights:
    torch.save({'ema_state_dict': ema.shadow, ...}, path)

    # Apply EMA for inference:
    ema.apply_shadow()
    model(...)
    ema.restore()
"""


class EMA:
    """Maintains an exponential moving average of model parameters.

    Args:
        model : nn.Module whose parameters are tracked.
        decay : smoothing factor (typical DDPM value: 0.9999).
    """

    def __init__(self, model, decay: float = 0.9999):
        self.model = model
        self.decay = decay
        self.shadow: dict = {}
        self._backup: dict = {}
        self._register()

    def _register(self) -> None:
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self) -> None:
        """Call after each optimizer.step()."""
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.shadow[name].mul_(self.decay).add_(
                    param.data, alpha=1.0 - self.decay
                )

    def apply_shadow(self) -> None:
        """Swap model weights with EMA weights (for inference). Call restore() after."""
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self._backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])

    def restore(self) -> None:
        """Restore original weights after apply_shadow()."""
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self._backup:
                param.data.copy_(self._backup[name])
        self._backup.clear()

    def load_shadow(self, shadow_dict: dict) -> None:
        """Load shadow weights from a checkpoint dict."""
        self.shadow = {k: v.clone() for k, v in shadow_dict.items()}
