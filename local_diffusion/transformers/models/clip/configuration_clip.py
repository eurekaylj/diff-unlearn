from huggingface_hub.dataclasses import strict
from transformers.configuration_utils import PreTrainedConfig
from transformers.utils import auto_docstring, logging
logger = logging.get_logger(__name__)

@strict
class CLIPTextConfig(PreTrainedConfig):
    model_type = 'clip_text_model'
    base_config_key = 'text_config'
    vocab_size: int = 49408
    hidden_size: int = 512
    intermediate_size: int = 2048
    projection_dim: int | None = 512
    num_hidden_layers: int = 12
    num_attention_heads: int = 8
    max_position_embeddings: int = 77
    hidden_act: str = 'quick_gelu'
    layer_norm_eps: float | None = 1e-05
    attention_dropout: int | float | None = 0.0
    initializer_range: float = 0.02
    initializer_factor: float | None = 1.0
    pad_token_id: int | None = 1
    bos_token_id: int | None = 49406
    eos_token_id: int | list[int] | None = 49407

    def validate_architecture(self):
        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError(f'The hidden size ({self.hidden_size}) is not a multiple of the number of attention heads ({self.num_attention_heads}).')

@strict
class CLIPVisionConfig(PreTrainedConfig):
    model_type = 'clip_vision_model'
    base_config_key = 'vision_config'
    hidden_size: int = 768
    intermediate_size: int = 3072
    projection_dim: int | None = 512
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    num_channels: int | None = 3
    image_size: int | None = 224
    patch_size: int | None = 32
    hidden_act: str = 'quick_gelu'
    layer_norm_eps: float | None = 1e-05
    attention_dropout: int | float | None = 0.0
    initializer_range: float = 0.02
    initializer_factor: float | None = 1.0

    def validate_architecture(self):
        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError(f'The hidden size ({self.hidden_size}) is not a multiple of the number of attention heads ({self.num_attention_heads}).')

@strict
class CLIPConfig(PreTrainedConfig):
    model_type = 'clip'
    sub_configs = {'text_config': CLIPTextConfig, 'vision_config': CLIPVisionConfig}
    text_config: dict | CLIPTextConfig | None = None
    vision_config: dict | CLIPVisionConfig | None = None
    projection_dim: int | None = 512
    logit_scale_init_value: float | int | None = 2.6592
    initializer_factor: float | None = 1.0

    def __post_init__(self, **kwargs):
        if self.text_config is None:
            text_config = {}
            logger.info('`text_config` is `None`. Initializing the `CLIPTextConfig` with default values.')
        elif isinstance(self.text_config, CLIPTextConfig):
            text_config = self.text_config.to_dict()
        else:
            text_config = self.text_config
        if self.vision_config is None:
            vision_config = {}
            logger.info('`vision_config` is `None`. initializing the `CLIPVisionConfig` with default values.')
        elif isinstance(self.vision_config, CLIPVisionConfig):
            vision_config = self.vision_config.to_dict()
        else:
            vision_config = self.vision_config
        text_config_dict = kwargs.pop('text_config_dict', None)
        vision_config_dict = kwargs.pop('vision_config_dict', None)
        if text_config_dict is not None:
            _text_config_dict = CLIPTextConfig(**text_config_dict).to_dict()
            for (key, value) in _text_config_dict.items():
                if key in text_config and value != text_config[key] and (key != 'transformers_version'):
                    if key in text_config_dict:
                        message = f'`{key}` is found in both `text_config_dict` and `text_config` but with different values. The value `text_config_dict["{key}"]` will be used instead.'
                    else:
                        message = f'`text_config_dict` is provided which will be used to initialize `CLIPTextConfig`. The value `text_config["{key}"]` will be overridden.'
                    logger.info(message)
            text_config.update(_text_config_dict)
        if vision_config_dict is not None:
            _vision_config_dict = CLIPVisionConfig(**vision_config_dict).to_dict()
            if 'id2label' in _vision_config_dict:
                _vision_config_dict['id2label'] = {str(key): value for (key, value) in _vision_config_dict['id2label'].items()}
            for (key, value) in _vision_config_dict.items():
                if key in vision_config and value != vision_config[key] and (key != 'transformers_version'):
                    if key in vision_config_dict:
                        message = f'`{key}` is found in both `vision_config_dict` and `vision_config` but with different values. The value `vision_config_dict["{key}"]` will be used instead.'
                    else:
                        message = f'`vision_config_dict` is provided which will be used to initialize `CLIPVisionConfig`. The value `vision_config["{key}"]` will be overridden.'
                    logger.info(message)
            vision_config.update(_vision_config_dict)
        self.text_config = CLIPTextConfig(**text_config)
        self.vision_config = CLIPVisionConfig(**vision_config)
        super().__post_init__(**kwargs)
__all__ = ['CLIPConfig', 'CLIPTextConfig', 'CLIPVisionConfig']
