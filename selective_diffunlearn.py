from contextlib import contextmanager
from dataclasses import dataclass
import math
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

@dataclass
class C06:
    resolution: int = 256
    diffusion_steps: int = 20
    start_step: int = 15
    latent_updates: int = 60
    latent_steps_per_round: int = 5
    latent_lr: float = 0.02
    epsilon: float | None = 16 / 255
    alignment_weight: float = 1.0
    attention_weight: float = 4.0
    visual_weight: float = 1.0
    inversion_guidance: float = 0.0
    denoising_guidance: float = 2.5
    attention_locations: tuple = ('down', 'mid', 'up')
    prompt_template: str = 'a photo of a {target}'
    zeta: float = 1e-06

    def f31(self):
        if not all((math.isfinite(v149) for v149 in (self.latent_lr, self.zeta, self.alignment_weight, self.attention_weight, self.visual_weight, self.inversion_guidance, self.denoising_guidance))):
            raise ValueError('Optimization settings must be finite.')
        if self.resolution <= 0 or self.resolution % 8:
            raise ValueError('Diffusion resolution must be a positive multiple of 8.')
        if not 0 <= self.start_step < self.diffusion_steps:
            raise ValueError('start_step must index the descending diffusion timetable.')
        if min(self.latent_updates, self.latent_steps_per_round) < 1:
            raise ValueError('Update counts and batch size must be positive.')
        if self.epsilon is not None and (not math.isfinite(self.epsilon) or not 0 <= self.epsilon <= 1):
            raise ValueError('epsilon must be in [0, 1].')
        if min(self.latent_lr, self.zeta) <= 0:
            raise ValueError('Learning rates and zeta must be positive.')
        if min(self.alignment_weight, self.attention_weight, self.visual_weight) < 0:
            raise ValueError('Loss weights must be nonnegative.')
        if not any((self.alignment_weight, self.attention_weight, self.visual_weight)):
            raise ValueError('At least one latent loss must be enabled.')
        if not self.attention_locations or set(self.attention_locations) - {'down', 'mid', 'up'}:
            raise ValueError('attention_locations must select down, mid and/or up.')
        if '{target}' not in self.prompt_template:
            raise ValueError('prompt_template must contain {target}.')

def f17(v111, v020, v037, v130=True):
    v104 = v111.clamp(0, 1) if v037 is None else (v020 + (v111 - v020).clamp(-v037, v037)).clamp(0, 1)
    return v111 + (v104 - v111).detach() if v130 else v104

def f22(v009, v159=1e-06):
    v009 = v009.detach()
    v081 = v009.amin(dim=(-2, -1), keepdim=True)
    v123 = v009.amax(dim=(-2, -1), keepdim=True) - v081
    return (v009 - v081) / (v123 + v159)

def f32(v048, v020, v076, v159=1e-06):
    v012 = 1 - v076
    v029 = v048 - v020
    v124 = (v012 * v029).square().sum(dim=(1, 2, 3))
    v031 = v012.sum(dim=(1, 2, 3)) + v159
    return (v124 / v031).mean() + v029.square().mean()

def f06(v048, v116):
    v041, v153 = v048.shape[-2:]
    v121 = (v116, int(v116 * v153 / v041)) if v041 <= v153 else (int(v116 * v041 / v153), v116)
    v115 = F.interpolate(v048, size=v121, mode='bicubic', align_corners=False, antialias=True)
    v144 = int(round((v121[0] - v116) / 2))
    v065 = int(round((v121[1] - v116) / 2))
    v115 = v115[..., v144:v144 + v116, v065:v065 + v116]
    v078 = v048.new_tensor((0.48145466, 0.4578275, 0.40821073))[None, :, None, None]
    v127 = v048.new_tensor((0.26862954, 0.26130258, 0.27577711))[None, :, None, None]
    return (v115 - v078) / v127

def f03(v082, v052, v143):
    v151 = F.normalize(v082.encode_image(v052).float(), dim=-1)
    v139 = F.normalize(v082.encode_text(v143).float(), dim=-1)
    return (1 - (v151 * v139).sum(dim=-1)).mean()

class C07:

    def __init__(self, v050):
        self.a07 = v050
        self.f23()

    def f23(self):
        self.a17 = {}
        self.a04 = {}

    def f20(self, v087, v102, v042):
        if v102.shape[0] != 2 * v042:
            raise ValueError('Target attention expects one image and two CFG branches.')
        v086 = v102.shape[1]
        v120 = math.isqrt(v086)
        if v120 * v120 != v086:
            raise ValueError('Expected square diffusion latent spatial positions.')
        v124 = v102[v042:, :, self.a07].mean(dim=(0, 2)).reshape(1, 1, v120, v120)
        self.a17[v087] = self.a17.get(v087, 0) + v124
        self.a04[v087] = self.a04.get(v087, 0) + 1

    def f02(self, v094):
        if not self.a17:
            raise RuntimeError('No target cross-attention was recorded.')
        if len(set(self.a04.values())) != 1:
            raise RuntimeError('Selected attention layers were not called equally often.')
        v075 = [F.interpolate(v149 / self.a04[v087], size=v094, mode='bilinear', align_corners=False) for v087, v149 in self.a17.items()]
        return sum(v075) / len(v075)

class C05:

    def __init__(self, v112, v087):
        self.a13, self.a10 = (v112, v087)

    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, temb=None, *args, **kwargs):
        v114 = hidden_states
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)
        v088 = hidden_states.ndim
        if v088 == 4:
            v014, v019, v043, v156 = hidden_states.shape
            hidden_states = hidden_states.view(v014, v019, v043 * v156).transpose(1, 2)
        v014, v066, v002 = encoder_hidden_states.shape
        attention_mask = attn.prepare_attention_mask(attention_mask, v066, v014)
        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)
        v110 = attn.to_q(hidden_states)
        if attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)
        v058 = attn.to_k(encoder_hidden_states)
        v149 = attn.to_v(encoder_hidden_states)
        v110, v058, v149 = (attn.head_to_batch_dim(v157) for v157 in (v110, v058, v149))
        v102 = attn.get_attention_scores(v110, v058, attention_mask)
        self.a13.f20(self.a10, v102, attn.heads)
        hidden_states = attn.batch_to_head_dim(torch.bmm(v102, v149))
        hidden_states = attn.to_out[1](attn.to_out[0](hidden_states))
        if v088 == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(v014, v019, v043, v156)
        if attn.residual_connection:
            hidden_states = hidden_states + v114
        return hidden_states / attn.rescale_output_factor

@contextmanager
def f21(v147, v112, v069):
    v093 = dict(v147.attn_processors)
    v103 = dict(v093)
    for v087 in v103:
        if '.attn2.' in v087 and v087.split('_', 1)[0] in v069:
            v103[v087] = C05(v112, v087)
    if all((v103[v056] is v093[v056] for v056 in v093)):
        raise ValueError('No SD cross-attention layers matched attention_locations.')
    try:
        v147.set_attn_processor(v103)
        yield
    finally:
        v147.set_attn_processor(v093)
        v112.f23()

def f26(v142, v105, v135):
    v106 = v142.encode(v105, truncation=True, max_length=v142.model_max_length)
    v136 = v142.encode(v135, add_special_tokens=False)
    v050 = []
    for v046 in range(1, len(v106) - len(v136)):
        if v106[v046:v046 + len(v136)] == v136:
            v050.extend(range(v046, v046 + len(v136)))
    if not v136 or not v050:
        raise ValueError(f'Target {v135!r} is absent or truncated in diffusion prompt {v105!r}.')
    return sorted(set(v050))

class C04:

    def __init__(self, v097, v024, v135):
        v024.f31()
        self.a11, self.a02 = (v097, v024)
        self.a06 = next(v097.unet.parameters()).device
        self.a12 = v024.prompt_template.format(target=v135)
        for v082 in (v097.unet, v097.vae, v097.text_encoder):
            v082.eval().requires_grad_(False)
        if getattr(v097.unet, 'is_gradient_checkpointing', False):
            raise ValueError('Stateful attention recording requires UNet gradient checkpointing disabled.')
        self.a16 = float(getattr(v097.vae.config, 'scaling_factor', 0.18215))
        if v097.scheduler.config.timestep_spacing != 'leading':
            raise ValueError("Use DDIM timestep_spacing='leading' for matched inversion/denoising.")
        v097.scheduler.set_timesteps(v024.diffusion_steps, device=self.a06)
        self.a18 = v097.scheduler.timesteps[v024.start_step:]
        self.a07 = f26(v097.tokenizer, self.a12, v135)
        with torch.no_grad():
            v047 = v097.tokenizer(['', self.a12], padding='max_length', truncation=True, max_length=v097.tokenizer.model_max_length, return_tensors='pt').input_ids.to(self.a06)
            self.a03 = v097.text_encoder(v047)[0].detach()

    def f14(self, v060, v140, v040):
        v052 = self.a11.scheduler.scale_model_input(torch.cat([v060, v060]), v140)
        v146, v022 = self.a11.unet(v052, v140, encoder_hidden_states=self.a03).sample.chunk(2)
        return v146 + v040 * (v022 - v146)

    @torch.no_grad()
    def f10(self, v020):
        v048 = F.interpolate(v020, size=(self.a02.resolution,) * 2, mode='bilinear', align_corners=False)
        v060 = self.a11.vae.encode(2 * v048 - 1).latent_dist.mode() * self.a16
        v118 = self.a11.scheduler
        if v118.config.prediction_type != 'epsilon':
            raise ValueError('This SD DDIM inversion currently supports epsilon prediction only.')
        v007 = v118.final_alpha_cumprod.to(v060)
        v101 = self.a18[-1]
        for v134 in reversed(self.a18):
            v089 = self.f14(v060, v101, self.a02.inversion_guidance)
            v006 = v118.alphas_cumprod[v134].to(v060)
            v158 = (v060 - (1 - v007).sqrt() * v089) / v007.sqrt()
            v060 = v006.sqrt() * v158 + (1 - v006).sqrt() * v089
            v101, v007 = (v134, v006)
        return v060.detach()

    def f07(self, v060, v121, v018=True):
        v112 = C07(self.a07)
        if v018:
            with f21(self.a11.unet, v112, self.a02.attention_locations):
                v111 = self.f01(v060, v121)
                v009 = v112.f02(v121)
            return (v111, v009)
        return (self.f01(v060, v121), None)

    def f01(self, v060, v121):
        for v134 in self.a18:
            v089 = self.f14(v060, v134, self.a02.denoising_guidance)
            v060 = self.a11.scheduler.step(v089, v134, v060, eta=0.0).prev_sample
        v111 = self.a11.vae.decode(v060 / self.a16).sample / 2 + 0.5
        return F.interpolate(v111, size=v121, mode='bilinear', align_corners=False)

    @torch.no_grad()
    def f09(self, v020):
        v013 = self.f10(v020)
        v111, v009 = self.f07(v013, v020.shape[-2:])
        return dict(base=v013.cpu(), delta=torch.zeros_like(v013).cpu(), mask=f22(v009, self.a02.zeta).cpu(), protected=f17(v111, v020, self.a02.epsilon, False).cpu(), optimizer=None, updates=0)

    def f30(self, v126, v020, v133, v143, v129):
        v024 = self.a02
        v133.eval().requires_grad_(False)
        v013, v076 = (v126['base'].to(self.a06), v126['mask'].to(self.a06))
        v029 = torch.nn.Parameter(v126['delta'].to(self.a06))
        v092 = torch.optim.Adam([v029], lr=v024.latent_lr)
        if v126['optimizer'] is not None:
            v092.load_state_dict(v126['optimizer'])
        v116 = int(v133.visual.input_resolution)
        v080 = []
        for v002 in range(v129):
            v092.zero_grad(set_to_none=True)
            v111, v009 = self.f07(v013 + v029, v020.shape[-2:])
            v048 = f17(v111, v020, v024.epsilon)
            v003 = f03(v133, f06(v048, v116), v143)
            v131 = (v076 * v009).sum() / (v076.sum() + v024.zeta)
            v151 = f32(v048, v020, v076, v024.zeta)
            v072 = v024.alignment_weight * v003 + v024.attention_weight * v131 + v024.visual_weight * v151
            if not torch.isfinite(v072):
                raise FloatingPointError('Non-finite latent objective.')
            v072.backward()
            if v029.grad is None or not torch.isfinite(v029.grad).all():
                raise FloatingPointError('Missing or non-finite latent gradient.')
            v092.step()
            v080.append(dict(alignment=float(v003.detach()), attention=float(v131.detach()), visual=float(v151.detach()), total=float(v072.detach())))
        with torch.no_grad():
            v111, v002 = self.f07(v013 + v029, v020.shape[-2:], v018=False)
            v108 = f17(v111, v020, v024.epsilon, False)
        v126.update(delta=v029.detach().cpu(), optimizer=f28(v092.state_dict()), protected=v108.cpu(), updates=v126['updates'] + v129)
        return (v126, v080)

def f28(v149):
    if torch.is_tensor(v149):
        return v149.detach().cpu()
    if isinstance(v149, dict):
        return {v058: f28(v054) for v058, v054 in v149.items()}
    if isinstance(v149, list):
        return [f28(v054) for v054 in v149]
    if isinstance(v149, tuple):
        return tuple((f28(v054) for v054 in v149))
    return v149

def f19(v096, v032='cpu'):
    with Image.open(v096) as v048:
        v008 = np.asarray(v048.convert('RGB'), dtype=np.float32).copy() / 255
    return torch.from_numpy(v008).permute(2, 0, 1).unsqueeze(0).to(v032)

def f24(v048, v020, v096, v037):
    v073 = torch.zeros_like(v020) if v037 is None else ((v020 - v037).clamp(0, 1) * 255).ceil()
    v044 = torch.full_like(v020, 255) if v037 is None else ((v020 + v037).clamp(0, 1) * 255).floor()
    v109 = (v048 * 255).round().maximum(v073).minimum(v044).to(torch.uint8)
    v008 = v109.squeeze(0).permute(1, 2, 0).cpu().numpy()
    Image.fromarray(v008).save(v096)
