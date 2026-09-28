import numpy as np
import torch
from cifar_selective_generation import f05, f13
from selective_diffunlearn import f17

def f27(v027, v050, v032):
    return torch.from_numpy(v027[v050].transpose(0, 3, 1, 2).copy()).float().to(v032) / 255

def f04(v035, v133, v027, v050, v135, args, v032):
    v117 = v050[np.linspace(0, len(v050) - 1, num=min(4, len(v050)), dtype=int)]
    v150 = {v058: [] for v058 in ('alignment', 'attention', 'visual')}
    v133.eval().requires_grad_(False)
    for v049 in v117:
        v020 = f27(v027.data, np.asarray([v049]), v032)
        with torch.no_grad():
            v126 = v035.f09(v020)
            v111, v009 = v035.f07(v126['base'].to(v032), v020.shape[-2:])
            v048 = f17(v111, v020, v035.a02.epsilon, False)
            v071 = v133(f05(v048, args.dataset, args.model))
            v055 = f13(v071, torch.tensor([v135], device=v032), v048, v020, v126['mask'].to(v032), v009, v035.a02)
            for v058 in v150:
                v150[v058].append(float(v055[v058].mean()))
    v079 = {v058: float(np.mean(v149)) for v058, v149 in v150.items()}
    if any((not np.isfinite(v149) or v149 <= 1e-08 for v149 in v079.values())):
        raise ValueError(f'Initial losses too small/nonfinite for calibration: {v079}')
    v155 = {v058: 1.0 / v149 for v058, v149 in v079.items()}
    v035.a02.alignment_weight = v155['alignment']
    v035.a02.attention_weight = v155['attention']
    v035.a02.visual_weight = v155['visual']
    v035.a02.f31()
    v154 = {v058: v079[v058] * v155[v058] for v058 in v079}
    assert max(v154.values()) / min(v154.values()) <= 10
    return dict(sample_indices=v117.tolist(), raw_initial_losses=v150, mean_initial_losses=v079, weights=v155, weighted_initial_losses=v154, rule='r01')
