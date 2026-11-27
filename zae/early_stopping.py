"""Validation-driven stopping, with checkpointable patience state."""
from dataclasses import dataclass, field
import math


@dataclass
class EarlyStopping:
    enabled: bool
    patience: int
    min_delta: float
    reference_psnr: float | None = field(default=None, init=False)
    bad_checks: int = field(default=0, init=False)
    stopped: bool = field(default=False, init=False)

    def __post_init__(self):
        if not isinstance(self.enabled, bool):
            raise ValueError("early_stopping.enabled must be true or false")
        if isinstance(self.patience, bool) or not isinstance(self.patience, int) or self.patience < 1:
            raise ValueError("early_stopping.patience must be a positive integer")
        if (isinstance(self.min_delta, bool) or not isinstance(self.min_delta, (int, float))
                or not math.isfinite(self.min_delta) or self.min_delta < 0):
            raise ValueError("early_stopping.min_delta must be finite and nonnegative")

    def update(self, psnr):
        if not math.isfinite(psnr):
            raise ValueError("Early stopping requires finite validation PSNR")
        if self.stopped:
            return True
        # Keep the last significant improvement as the reference so small gains can accumulate.
        if self.reference_psnr is None or psnr > self.reference_psnr + self.min_delta:
            self.reference_psnr = psnr
            self.bad_checks = 0
        else:
            self.bad_checks += 1
        self.stopped = self.enabled and self.bad_checks >= self.patience
        return self.stopped

    def state_dict(self):
        return {"reference_psnr": self.reference_psnr,
                "bad_checks": self.bad_checks, "stopped": self.stopped}

    def load_state_dict(self, state):
        self.reference_psnr = state["reference_psnr"]
        self.bad_checks = state["bad_checks"]
        self.stopped = state["stopped"]
