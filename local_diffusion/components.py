import json
from pathlib import Path
from types import SimpleNamespace
import torch
from safetensors.torch import load_file
from transformers import CLIPTokenizer
from .diffusers.models.unets.unet_2d_condition import UNet2DConditionModel
from .diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL
from .diffusers.schedulers.scheduling_ddim import DDIMScheduler
from .transformers.models.clip.modeling_clip import CLIPTextModel
from .transformers.models.clip.configuration_clip import CLIPTextConfig

def f11(v083, v032='cpu'):
    v039 = Path(v083)

    def f18(v113):
        return json.loads((v039 / v113).read_text())
    v147 = UNet2DConditionModel.from_config(f18('unet/config.json'))
    v148 = AutoencoderKL.from_config(f18('vae/config.json'))
    v138 = CLIPTextModel(CLIPTextConfig.from_dict(f18('text_encoder/config.json')))
    for v082, v038 in ((v147, 'unet/diffusion_pytorch_model.safetensors'), (v148, 'vae/diffusion_pytorch_model.safetensors'), (v138, 'text_encoder/model.safetensors')):
        v082.load_state_dict(load_file(str(v039 / v038), device='cpu'), strict=True)
        v082.eval().requires_grad_(False).to(v032)
    v118 = DDIMScheduler.from_config(f18('scheduler/scheduler_config.json'), clip_sample=False, timestep_spacing='leading')
    v142 = CLIPTokenizer.from_pretrained(str(v039 / 'tokenizer'), local_files_only=True)
    return SimpleNamespace(unet=v147, vae=v148, text_encoder=v138, tokenizer=v142, scheduler=v118)
