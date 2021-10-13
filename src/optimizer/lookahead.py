from typing import TYPE_CHECKING, Any, Callable, Optional

import torch
import torch.optim
from torch.optim.optimizer import Optimizer

if TYPE_CHECKING:
    from torch.optim.optimizer import _params_t
else:
    _params_t = Any


class LookAhead(torch.optim.Optimizer):
    def __init__(self, params: _params_t, base_optimizer: Optimizer, period: int = 10, slow_lr: float = 0.2, period2: int = 1042):
        defaults = dict(period = period, period2 = period2, slow_lr = slow_lr, base_optimizer = base_optimizer)
        super().__init__(params, defaults)

    def step(self):
        if "k" not in self.state:
            self.state["k"] = torch.tensor([0], dtype = torch.long)
        k = self.state["k"].item()
        if "k2" not in self.state:
            self.state["k2"] = torch.tensor([0], dtype = torch.long)
        k2 = self.state["k2"].item()

        loss = self.defaults["base_optimizer"].step()

        if k % self.defaults["period"] == 0 or k2 % self.defaults["period2"] == 0:
            for group in self.param_groups:
                for fast_param in group["params"]:
                    param_state = self.state[fast_param]
                    if "slow_param" not in param_state:
                        param_state["slow_param"] = torch.zeros_like(fast_param.data)
                        param_state["slow_param"].copy_(fast_param.data)
                    slow = param_state["slow_param"]
                    slow += (fast_param.data - slow) * self.defaults["slow_lr"]
                    fast_param.data.copy_(slow)

        self.state["k"] += 1
        self.state["k2"] += 1
            

        return loss
            
