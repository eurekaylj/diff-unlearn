from dataclasses import dataclass
from diffusers.utils import BaseOutput

@dataclass
class AutoencoderKLOutput(BaseOutput):
    latent_dist: 'DiagonalGaussianDistribution'

@dataclass
class Transformer2DModelOutput(BaseOutput):
    sample: 'torch.Tensor'
