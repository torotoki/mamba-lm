import torch
from mamba_ssm import Mamba as MambaBlockOfficial

from ..mamba import MambaBlock, MambaConfig

batch, length, dim = 2, 64, 16
torch.seed(42)
x = torch.randn(batch, length, dim).to("cuda")
print("Mamba Block Official:")
torch.seed(42)
model = MambaBlockOfficial(
    # This module uses roughly 3 * expand * d_model^2 parameters
    d_model=dim, # Model dimension d_model
    d_state=16,  # SSM state expansion factor
    d_conv=4,    # Local convolution width
    expand=2,    # Block expansion factor
).to("cuda")
y = model(x)
assert y.shape == x.shape
print(y)

print("Mamba Block:")
torch.seed(42)
config = MambaConfig(
    d_model=dim,
    d_state=16,
    d_conv=4,
    expand=2
)
model = MambaBlock(config).to("cuda")
y = model(x)
assert y.shape == x.shape
print(y)
