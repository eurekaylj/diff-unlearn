import math
import torch
import torch.nn.functional as F
from selective_diffunlearn import C04, C07, f17, f21, f28, f26

def f05(v048, v028, v082):
    if v082 in ('resnet18', 'resnet50', 'vgg16_bn', 'densenet121'):
        if v048.shape[-2:] != (32, 32):
            v048 = F.interpolate(v048, size=(32, 32), mode='bilinear', align_corners=False)
        return v048 - 0.5
    if v082 == 'vit_b16':
        v048 = F.interpolate(v048, size=(248, 248), mode='bicubic', align_corners=False, antialias=True)
        v048 = v048[..., 12:236, 12:236].clamp(0, 1)
        v078, v127 = ((0.5,) * 3, (0.5,) * 3)
    else:
        if v048.shape[-2:] != (32, 32):
            v048 = F.interpolate(v048, size=(32, 32), mode='bilinear', align_corners=False)
        v078, v127 = ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.201)) if v028 == 'cifar10' else ((0.507, 0.4865, 0.4409), (0.2673, 0.2564, 0.2761))
    return (v048 - v048.new_tensor(v078)[None, :, None, None]) / v048.new_tensor(v127)[None, :, None, None]

def f13(v071, v059, v048, v020, v076, v009, v024):
    v004 = F.cross_entropy(v071.float(), v059, reduction='none')
    v132 = (v076 * v009).sum((1, 2, 3)) / (v076.sum((1, 2, 3)) + v024.zeta)
    v012 = 1 - v076
    v029 = v048 - v020
    v151 = (v012 * v029).square().sum((1, 2, 3)) / (v012.sum((1, 2, 3)) + v024.zeta)
    v151 = v151 + v029.square().mean((1, 2, 3))
    v145 = v024.alignment_weight * v004 + v024.attention_weight * v132 + v024.visual_weight * v151
    return dict(alignment=v004, attention=v132, visual=v151, total=v145)

class C01(C07):

    def __init__(self, v050, v015):
        super().__init__(v050)
        self.a01 = v015

    def f20(self, v087, v102, v042):
        v014 = self.a01
        if v102.shape[0] != 2 * v014 * v042:
            raise ValueError('Expected unconditional batch followed by conditional batch')
        v099, v143 = v102.shape[1:]
        v120 = math.isqrt(v099)
        if v120 * v120 != v099:
            raise ValueError('Expected square latent positions')
        v023 = v102.reshape(2, v014, v042, v099, v143)[1]
        v119 = v023.index_select(-1, torch.tensor(self.a07, device=v102.device))
        v124 = v119.float().mean(dim=(1, 3)).reshape(v014, 1, v120, v120)
        self.a17[v087] = self.a17.get(v087, 0) + v124
        self.a04[v087] = self.a04.get(v087, 0) + 1

class C02(C04):

    def __init__(self, v097, v024, v135, v028, v084):
        v024.f31()
        self.a11, self.a02 = (v097, v024)
        self.a06 = next(v097.unet.parameters()).device
        self.a12 = v024.prompt_template.format(target=v135)
        for v085 in (v097.unet, v097.vae, v097.text_encoder):
            v085.eval().requires_grad_(False)
        if getattr(v097.unet, 'is_gradient_checkpointing', False):
            raise ValueError('Disable UNet gradient checkpointing for attention recording')
        self.a16 = float(getattr(v097.vae.config, 'scaling_factor', 0.18215))
        if v097.scheduler.config.timestep_spacing != 'leading':
            raise ValueError('Expected leading DDIM timetable')
        v097.scheduler.set_timesteps(v024.diffusion_steps, device=self.a06)
        self.a18 = v097.scheduler.timesteps[v024.start_step:]
        self.a07 = f26(v097.tokenizer, self.a12, v135)
        with torch.no_grad():
            v047 = v097.tokenizer(['', self.a12], padding='max_length', truncation=True, max_length=v097.tokenizer.model_max_length, return_tensors='pt').input_ids
            v137 = next(v097.text_encoder.parameters()).device
            self.a03 = v097.text_encoder(v047.to(v137))[0].detach().to(self.a06)
        self.a05, self.a09 = (v028, v084)

    def f14(self, v060, v140, v040):
        v014 = len(v060)
        v026 = torch.cat([self.a03[:1].expand(v014, -1, -1), self.a03[1:].expand(v014, -1, -1)])
        v052 = self.a11.scheduler.scale_model_input(torch.cat([v060, v060]), v140)
        v034 = next(self.a11.unet.parameters()).dtype
        v146, v022 = self.a11.unet(v052.to(v034), v140, encoder_hidden_states=v026.to(v034)).sample.float().chunk(2)
        return v146 + v040 * (v022 - v146)

    def f07(self, v060, v121, v018=True):
        if not v018:
            return (self.f01(v060, v121), None)
        v112 = C01(self.a07, len(v060))
        with f21(self.a11.unet, v112, self.a02.attention_locations):
            v111 = self.f01(v060, v121)
            v009 = v112.f02(v121)
        return (v111, v009)

    def f30(self, v126, v020, v133, v059, v129):
        v024 = self.a02
        v133.eval().requires_grad_(False)
        v013, v076 = (v126['base'].to(self.a06), v126['mask'].to(self.a06))
        v029 = torch.nn.Parameter(v126['delta'].to(self.a06))
        v092 = torch.optim.Adam([v029], lr=v024.latent_lr)
        if v126['optimizer'] is not None:
            v092.load_state_dict(v126['optimizer'])
        v045 = []
        for v002 in range(v129):
            v092.zero_grad(set_to_none=True)
            v111, v009 = self.f07(v013 + v029, v020.shape[-2:])
            v048 = f17(v111, v020, v024.epsilon)
            v071 = v133(f05(v048, self.a05, self.a09))
            v080 = f13(v071, v059, v048, v020, v076, v009, v024)
            v072 = v080['total'].sum()
            if not torch.isfinite(v072):
                raise FloatingPointError('Non-finite generation loss')
            v072.backward()
            if v029.grad is None or not torch.isfinite(v029.grad).all():
                raise FloatingPointError('Invalid latent gradient')
            v092.step()
            v045.append({v058: float(v149.detach().mean()) for v058, v149 in v080.items()})
        with torch.no_grad():
            v111, v002 = self.f07(v013 + v029, v020.shape[-2:], v018=False)
            v108 = f17(v111, v020, v024.epsilon, False)
        v126.update(delta=v029.detach().cpu(), optimizer=f28(v092.state_dict()), protected=v108.cpu(), updates=v126['updates'] + v129)
        return (v126, v045)
