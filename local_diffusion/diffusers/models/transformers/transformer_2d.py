from typing import Any
import torch
import torch.nn.functional as F
from torch import nn
from diffusers.configuration_utils import LegacyConfigMixin, register_to_config
from diffusers.utils import deprecate, logging
from local_diffusion.diffusers.models.attention import BasicTransformerBlock
from local_diffusion.diffusers.models.embeddings import ImagePositionalEmbeddings, PatchEmbed, PixArtAlphaTextProjection
from local_diffusion.diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.modeling_utils import LegacyModelMixin
from local_diffusion.diffusers.models.normalization import AdaLayerNormSingle
logger = logging.get_logger(__name__)

class Transformer2DModelOutput(Transformer2DModelOutput):

    def __init__(self, *args, **kwargs):
        deprecation_message = 'Importing `Transformer2DModelOutput` from `diffusers.models.transformer_2d` is deprecated and this will be removed in a future version. Please use `from diffusers.models.modeling_outputs import Transformer2DModelOutput`, instead.'
        deprecate('Transformer2DModelOutput', '1.0.0', deprecation_message)
        super().__init__(*args, **kwargs)

class Transformer2DModel(LegacyModelMixin, LegacyConfigMixin):
    _supports_gradient_checkpointing = True
    _no_split_modules = ['BasicTransformerBlock']
    _skip_layerwise_casting_patterns = ['latent_image_embedding', 'norm']

    @register_to_config
    def __init__(self, num_attention_heads: int=16, attention_head_dim: int=88, in_channels: int | None=None, out_channels: int | None=None, num_layers: int=1, dropout: float=0.0, norm_num_groups: int=32, cross_attention_dim: int | None=None, attention_bias: bool=False, sample_size: int | None=None, num_vector_embeds: int | None=None, patch_size: int | None=None, activation_fn: str='geglu', num_embeds_ada_norm: int | None=None, use_linear_projection: bool=False, only_cross_attention: bool=False, double_self_attention: bool=False, upcast_attention: bool=False, norm_type: str='layer_norm', norm_elementwise_affine: bool=True, norm_eps: float=1e-05, attention_type: str='default', caption_channels: int=None, interpolation_scale: float=None, use_additional_conditions: bool | None=None):
        super().__init__()
        if patch_size is not None:
            if norm_type not in ['ada_norm', 'ada_norm_zero', 'ada_norm_single']:
                raise NotImplementedError(f"Forward pass is not implemented when `patch_size` is not None and `norm_type` is '{norm_type}'.")
            elif norm_type in ['ada_norm', 'ada_norm_zero'] and num_embeds_ada_norm is None:
                raise ValueError(f'When using a `patch_size` and this `norm_type` ({norm_type}), `num_embeds_ada_norm` cannot be None.')
        self.is_input_continuous = in_channels is not None and patch_size is None
        self.is_input_vectorized = num_vector_embeds is not None
        self.is_input_patches = in_channels is not None and patch_size is not None
        if self.is_input_continuous and self.is_input_vectorized:
            raise ValueError(f'Cannot define both `in_channels`: {in_channels} and `num_vector_embeds`: {num_vector_embeds}. Make sure that either `in_channels` or `num_vector_embeds` is None.')
        elif self.is_input_vectorized and self.is_input_patches:
            raise ValueError(f'Cannot define both `num_vector_embeds`: {num_vector_embeds} and `patch_size`: {patch_size}. Make sure that either `num_vector_embeds` or `num_patches` is None.')
        elif not self.is_input_continuous and (not self.is_input_vectorized) and (not self.is_input_patches):
            raise ValueError(f'Has to define `in_channels`: {in_channels}, `num_vector_embeds`: {num_vector_embeds}, or patch_size: {patch_size}. Make sure that `in_channels`, `num_vector_embeds` or `num_patches` is not None.')
        if norm_type == 'layer_norm' and num_embeds_ada_norm is not None:
            deprecation_message = f"The configuration file of this model: {self.__class__} is outdated. `norm_type` is either not set or incorrectly set to `'layer_norm'`. Make sure to set `norm_type` to `'ada_norm'` in the config. Please make sure to update the config accordingly as leaving `norm_type` might led to incorrect results in future versions. If you have downloaded this checkpoint from the Hugging Face Hub, it would be very nice if you could open a Pull request for the `transformer/config.json` file"
            deprecate('norm_type!=num_embeds_ada_norm', '1.0.0', deprecation_message, standard_warn=False)
            norm_type = 'ada_norm'
        self.use_linear_projection = use_linear_projection
        self.interpolation_scale = interpolation_scale
        self.caption_channels = caption_channels
        self.num_attention_heads = num_attention_heads
        self.attention_head_dim = attention_head_dim
        self.inner_dim = self.config.num_attention_heads * self.config.attention_head_dim
        self.in_channels = in_channels
        self.out_channels = in_channels if out_channels is None else out_channels
        self.gradient_checkpointing = False
        if use_additional_conditions is None:
            if norm_type == 'ada_norm_single' and sample_size == 128:
                use_additional_conditions = True
            else:
                use_additional_conditions = False
        self.use_additional_conditions = use_additional_conditions
        if self.is_input_continuous:
            self._init_continuous_input(norm_type=norm_type)
        elif self.is_input_vectorized:
            self._init_vectorized_inputs(norm_type=norm_type)
        elif self.is_input_patches:
            self._init_patched_inputs(norm_type=norm_type)

    def _init_continuous_input(self, norm_type):
        self.norm = torch.nn.GroupNorm(num_groups=self.config.norm_num_groups, num_channels=self.in_channels, eps=1e-06, affine=True)
        if self.use_linear_projection:
            self.proj_in = torch.nn.Linear(self.in_channels, self.inner_dim)
        else:
            self.proj_in = torch.nn.Conv2d(self.in_channels, self.inner_dim, kernel_size=1, stride=1, padding=0)
        self.transformer_blocks = nn.ModuleList([BasicTransformerBlock(self.inner_dim, self.config.num_attention_heads, self.config.attention_head_dim, dropout=self.config.dropout, cross_attention_dim=self.config.cross_attention_dim, activation_fn=self.config.activation_fn, num_embeds_ada_norm=self.config.num_embeds_ada_norm, attention_bias=self.config.attention_bias, only_cross_attention=self.config.only_cross_attention, double_self_attention=self.config.double_self_attention, upcast_attention=self.config.upcast_attention, norm_type=norm_type, norm_elementwise_affine=self.config.norm_elementwise_affine, norm_eps=self.config.norm_eps, attention_type=self.config.attention_type) for _ in range(self.config.num_layers)])
        if self.use_linear_projection:
            self.proj_out = torch.nn.Linear(self.inner_dim, self.out_channels)
        else:
            self.proj_out = torch.nn.Conv2d(self.inner_dim, self.out_channels, kernel_size=1, stride=1, padding=0)

    def _init_vectorized_inputs(self, norm_type):
        assert self.config.sample_size is not None, 'Transformer2DModel over discrete input must provide sample_size'
        assert self.config.num_vector_embeds is not None, 'Transformer2DModel over discrete input must provide num_embed'
        self.height = self.config.sample_size
        self.width = self.config.sample_size
        self.num_latent_pixels = self.height * self.width
        self.latent_image_embedding = ImagePositionalEmbeddings(num_embed=self.config.num_vector_embeds, embed_dim=self.inner_dim, height=self.height, width=self.width)
        self.transformer_blocks = nn.ModuleList([BasicTransformerBlock(self.inner_dim, self.config.num_attention_heads, self.config.attention_head_dim, dropout=self.config.dropout, cross_attention_dim=self.config.cross_attention_dim, activation_fn=self.config.activation_fn, num_embeds_ada_norm=self.config.num_embeds_ada_norm, attention_bias=self.config.attention_bias, only_cross_attention=self.config.only_cross_attention, double_self_attention=self.config.double_self_attention, upcast_attention=self.config.upcast_attention, norm_type=norm_type, norm_elementwise_affine=self.config.norm_elementwise_affine, norm_eps=self.config.norm_eps, attention_type=self.config.attention_type) for _ in range(self.config.num_layers)])
        self.norm_out = nn.LayerNorm(self.inner_dim)
        self.out = nn.Linear(self.inner_dim, self.config.num_vector_embeds - 1)

    def _init_patched_inputs(self, norm_type):
        assert self.config.sample_size is not None, 'Transformer2DModel over patched input must provide sample_size'
        self.height = self.config.sample_size
        self.width = self.config.sample_size
        self.patch_size = self.config.patch_size
        interpolation_scale = self.config.interpolation_scale if self.config.interpolation_scale is not None else max(self.config.sample_size // 64, 1)
        self.pos_embed = PatchEmbed(height=self.config.sample_size, width=self.config.sample_size, patch_size=self.config.patch_size, in_channels=self.in_channels, embed_dim=self.inner_dim, interpolation_scale=interpolation_scale)
        self.transformer_blocks = nn.ModuleList([BasicTransformerBlock(self.inner_dim, self.config.num_attention_heads, self.config.attention_head_dim, dropout=self.config.dropout, cross_attention_dim=self.config.cross_attention_dim, activation_fn=self.config.activation_fn, num_embeds_ada_norm=self.config.num_embeds_ada_norm, attention_bias=self.config.attention_bias, only_cross_attention=self.config.only_cross_attention, double_self_attention=self.config.double_self_attention, upcast_attention=self.config.upcast_attention, norm_type=norm_type, norm_elementwise_affine=self.config.norm_elementwise_affine, norm_eps=self.config.norm_eps, attention_type=self.config.attention_type) for _ in range(self.config.num_layers)])
        if self.config.norm_type != 'ada_norm_single':
            self.norm_out = nn.LayerNorm(self.inner_dim, elementwise_affine=False, eps=1e-06)
            self.proj_out_1 = nn.Linear(self.inner_dim, 2 * self.inner_dim)
            self.proj_out_2 = nn.Linear(self.inner_dim, self.config.patch_size * self.config.patch_size * self.out_channels)
        elif self.config.norm_type == 'ada_norm_single':
            self.norm_out = nn.LayerNorm(self.inner_dim, elementwise_affine=False, eps=1e-06)
            self.scale_shift_table = nn.Parameter(torch.randn(2, self.inner_dim) / self.inner_dim ** 0.5)
            self.proj_out = nn.Linear(self.inner_dim, self.config.patch_size * self.config.patch_size * self.out_channels)
        self.adaln_single = None
        if self.config.norm_type == 'ada_norm_single':
            self.adaln_single = AdaLayerNormSingle(self.inner_dim, use_additional_conditions=self.use_additional_conditions)
        self.caption_projection = None
        if self.caption_channels is not None:
            self.caption_projection = PixArtAlphaTextProjection(in_features=self.caption_channels, hidden_size=self.inner_dim)

    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor | None=None, timestep: torch.LongTensor | None=None, added_cond_kwargs: dict[str, torch.Tensor]=None, class_labels: torch.LongTensor | None=None, cross_attention_kwargs: dict[str, Any]=None, attention_mask: torch.Tensor | None=None, encoder_attention_mask: torch.Tensor | None=None, return_dict: bool=True):
        if cross_attention_kwargs is not None:
            if cross_attention_kwargs.get('scale', None) is not None:
                logger.warning('Passing `scale` to `cross_attention_kwargs` is deprecated. `scale` will be ignored.')
        if attention_mask is not None and attention_mask.ndim == 2:
            attention_mask = (1 - attention_mask.to(hidden_states.dtype)) * -10000.0
            attention_mask = attention_mask.unsqueeze(1)
        if encoder_attention_mask is not None and encoder_attention_mask.ndim == 2:
            encoder_attention_mask = (1 - encoder_attention_mask.to(hidden_states.dtype)) * -10000.0
            encoder_attention_mask = encoder_attention_mask.unsqueeze(1)
        if self.is_input_continuous:
            (batch_size, _, height, width) = hidden_states.shape
            residual = hidden_states
            (hidden_states, inner_dim) = self._operate_on_continuous_inputs(hidden_states)
        elif self.is_input_vectorized:
            hidden_states = self.latent_image_embedding(hidden_states)
        elif self.is_input_patches:
            (height, width) = (hidden_states.shape[-2] // self.patch_size, hidden_states.shape[-1] // self.patch_size)
            (hidden_states, encoder_hidden_states, timestep, embedded_timestep) = self._operate_on_patched_inputs(hidden_states, encoder_hidden_states, timestep, added_cond_kwargs)
        for block in self.transformer_blocks:
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                hidden_states = self._gradient_checkpointing_func(block, hidden_states, attention_mask, encoder_hidden_states, encoder_attention_mask, timestep, cross_attention_kwargs, class_labels)
            else:
                hidden_states = block(hidden_states, attention_mask=attention_mask, encoder_hidden_states=encoder_hidden_states, encoder_attention_mask=encoder_attention_mask, timestep=timestep, cross_attention_kwargs=cross_attention_kwargs, class_labels=class_labels)
        if self.is_input_continuous:
            output = self._get_output_for_continuous_inputs(hidden_states=hidden_states, residual=residual, batch_size=batch_size, height=height, width=width, inner_dim=inner_dim)
        elif self.is_input_vectorized:
            output = self._get_output_for_vectorized_inputs(hidden_states)
        elif self.is_input_patches:
            output = self._get_output_for_patched_inputs(hidden_states=hidden_states, timestep=timestep, class_labels=class_labels, embedded_timestep=embedded_timestep, height=height, width=width)
        if not return_dict:
            return (output,)
        return Transformer2DModelOutput(sample=output)

    def _operate_on_continuous_inputs(self, hidden_states):
        (batch, _, height, width) = hidden_states.shape
        hidden_states = self.norm(hidden_states)
        if not self.use_linear_projection:
            hidden_states = self.proj_in(hidden_states)
            inner_dim = hidden_states.shape[1]
            hidden_states = hidden_states.permute(0, 2, 3, 1).reshape(batch, height * width, inner_dim)
        else:
            inner_dim = hidden_states.shape[1]
            hidden_states = hidden_states.permute(0, 2, 3, 1).reshape(batch, height * width, inner_dim)
            hidden_states = self.proj_in(hidden_states)
        return (hidden_states, inner_dim)

    def _operate_on_patched_inputs(self, hidden_states, encoder_hidden_states, timestep, added_cond_kwargs):
        batch_size = hidden_states.shape[0]
        hidden_states = self.pos_embed(hidden_states)
        embedded_timestep = None
        if self.adaln_single is not None:
            if self.use_additional_conditions and added_cond_kwargs is None:
                raise ValueError('`added_cond_kwargs` cannot be None when using additional conditions for `adaln_single`.')
            (timestep, embedded_timestep) = self.adaln_single(timestep, added_cond_kwargs, batch_size=batch_size, hidden_dtype=hidden_states.dtype)
        if self.caption_projection is not None:
            encoder_hidden_states = self.caption_projection(encoder_hidden_states)
            encoder_hidden_states = encoder_hidden_states.view(batch_size, -1, hidden_states.shape[-1])
        return (hidden_states, encoder_hidden_states, timestep, embedded_timestep)

    def _get_output_for_continuous_inputs(self, hidden_states, residual, batch_size, height, width, inner_dim):
        if not self.use_linear_projection:
            hidden_states = hidden_states.reshape(batch_size, height, width, inner_dim).permute(0, 3, 1, 2).contiguous()
            hidden_states = self.proj_out(hidden_states)
        else:
            hidden_states = self.proj_out(hidden_states)
            hidden_states = hidden_states.reshape(batch_size, height, width, inner_dim).permute(0, 3, 1, 2).contiguous()
        output = hidden_states + residual
        return output

    def _get_output_for_vectorized_inputs(self, hidden_states):
        hidden_states = self.norm_out(hidden_states)
        logits = self.out(hidden_states)
        logits = logits.permute(0, 2, 1)
        output = F.log_softmax(logits.double(), dim=1).float()
        return output

    def _get_output_for_patched_inputs(self, hidden_states, timestep, class_labels, embedded_timestep, height=None, width=None):
        if self.config.norm_type != 'ada_norm_single':
            conditioning = self.transformer_blocks[0].norm1.emb(timestep, class_labels, hidden_dtype=hidden_states.dtype)
            (shift, scale) = self.proj_out_1(F.silu(conditioning)).chunk(2, dim=1)
            hidden_states = self.norm_out(hidden_states) * (1 + scale[:, None]) + shift[:, None]
            hidden_states = self.proj_out_2(hidden_states)
        elif self.config.norm_type == 'ada_norm_single':
            (shift, scale) = (self.scale_shift_table[None] + embedded_timestep[:, None]).chunk(2, dim=1)
            hidden_states = self.norm_out(hidden_states)
            hidden_states = hidden_states * (1 + scale) + shift
            hidden_states = self.proj_out(hidden_states)
            hidden_states = hidden_states.squeeze(1)
        if self.adaln_single is None:
            height = width = int(hidden_states.shape[1] ** 0.5)
        hidden_states = hidden_states.reshape(shape=(-1, height, width, self.patch_size, self.patch_size, self.out_channels))
        hidden_states = torch.einsum('nhwpqc->nchpwq', hidden_states)
        output = hidden_states.reshape(shape=(-1, self.out_channels, height * self.patch_size, width * self.patch_size))
        return output
