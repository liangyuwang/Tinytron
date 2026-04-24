import torch
import torch.nn as nn

from tinytron.model.config import ModelConfig

class LayerNorm(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.device = torch.cuda.current_device()
        self.variance_epsilon = 1e-6
        self.hidden_size = config.hidden_size
        self.weight = nn.Parameter(torch.ones(self.hidden_size, device=self.device))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype
        x = x.to(torch.float32)
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * x.to(input_dtype)
