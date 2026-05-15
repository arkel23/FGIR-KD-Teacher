class ViTConfig():
    def __init__(self,
                 model_name: str = 'vit_b16',
                 debugging: bool = None,
                 image_size: int = None,
                 patch_size: tuple = None,
                 patch_stride: int = None,
                 load_fc_layer: bool = None,
                 classifier: str = None,
                 num_classes: int = None,
                 num_channels: int = None,

                 attention: str = None,
                 pos_embedding_type: str = None,
                 hidden_size: int = None,
                 intermediate_size: int = None,
                 num_attention_heads: int = None,
                 num_hidden_layers: int = None,
                 encoder_norm: bool = None,

                 representation_size: int = None,
                 attention_probs_dropout_prob: float = None,
                 hidden_dropout_prob: float = None,
                 sd: float = None,
                 layer_norm_eps: float = None,
                 hidden_act: str = None,

                 url: str = None,
                 print_attr: bool = False,
                 **kwargs):
        super().__init__(**kwargs)
        self.model_name = model_name
        default = CONFIGS[model_name]

        input_args = locals()
        for k, v in input_args.items():
            if k in default['config'].keys():
                setattr(self, k, v if v is not None else default['config'][k])

        self.calc_dims()
        if print_attr:
            print(vars(self))

    def as_tuple(self, x):
        return x if isinstance(x, tuple) else (x, x)

    def calc_dims(self):
        self.fh, self.fw = self.as_tuple(self.patch_size)  # patch sizes

        if not self.patch_stride:
            self.patch_stride = self.fh

        # number of patches
        self.gh = (((self.image_size - self.fh) // self.patch_stride) + 1)
        self.gw = (((self.image_size - self.fw) // self.patch_stride) + 1)

        self.seq_len = self.gh * self.gw  # sequence length individual image
        assert(self.gh ** 2 == self.seq_len), 'Only accept same H=W features'

        if 'cls' in self.classifier:
            self.seq_len += 1

        self.patch_equivalent = self.image_size // self.gh

        self.representation_size = self.hidden_size

    def __repr__(self):
        return str(vars(self))

    def __str__(self):
        return str(vars(self))


def get_base_config():
    """Base ViT config ViT"""
    return dict(
        debugging=False,
        image_size=224,
        patch_size=(16, 16),
        patch_stride=None,
        load_fc_layer=True,
        classifier='cls',
        num_classes=1000,
        num_channels=3,

        attention='vanilla',
        pos_embedding_type='learned',
        hidden_size=768,
        intermediate_size=3072,
        num_attention_heads=12,
        num_hidden_layers=12,
        encoder_norm=True,

        representation_size=768,
        attention_probs_dropout_prob=0.0,
        hidden_dropout_prob=0.1,
        sd=0.0,
        layer_norm_eps=1e-12,
        hidden_act='gelu',

       url=None,
    )


def get_b16_config():
    """Returns the ViT-B/16 configuration."""
    config = get_base_config()
    config.update(dict(
        url='https://storage.googleapis.com/vit_models/imagenet21k/ViT-B_16.npz'))
    return config


def get_b32_config():
    """Returns the ViT-B/32 configuration."""
    config = get_base_config()
    config.update(dict(
        patch_size=(32, 32),
        url='https://storage.googleapis.com/vit_models/imagenet21k/ViT-B_32.npz'))
    return config


def get_b8_config():
    """Returns the ViT-B/8 configuration."""
    config = get_base_config()
    config.update(dict(
        patch_size=(8, 8),
        url='https://storage.googleapis.com/vit_models/imagenet21k/ViT-B_8.npz'))
    return config


def get_l16_config():
    """Returns the ViT-L/16 configuration."""
    config = get_base_config()
    config.update(dict(
        hidden_size=1024,
        intermediate_size=4096,
        num_attention_heads=16,
        num_hidden_layers=24,
        representation_size=1024,
        url='https://storage.googleapis.com/vit_models/imagenet21k/ViT-L_16.npz'
    ))
    return config


def get_l32_config():
    """Returns the ViT-L/32 configuration."""
    config = get_l16_config()
    config.update(dict(
        patch_size=(32, 32),
        url='https://storage.googleapis.com/vit_models/imagenet21k/ViT-L_32.npz'))
    return config


def get_h14_config():
    """Returns the ViT-H/14 configuration."""
    config = get_base_config()
    config.update(dict(
        patch_size=(14, 14),
        hidden_size=1280,
        intermediate_size=5120,
        num_attention_heads=16,
        num_hidden_layers=32,
        representation_size=1280,
        url='https://storage.googleapis.com/vit_models/imagenet21k/ViT-H_14.npz'
    ))
    return config


def get_s16_config():
    """Returns the ViT-S/16 configuration."""
    config = get_base_config()
    config.update(dict(
        hidden_size=384,
        intermediate_size=1536,
        num_attention_heads=6,
        representation_size=384,
    ))
    return config


def get_s32_config():
    """Returns the ViT-S/32 configuration."""
    config = get_s16_config()
    config.update(dict(patch_size=(32, 32)))
    return config


def get_s8_config():
    """Returns the ViT-S/8 configuration."""
    config = get_s16_config()
    config.update(dict(patch_size=(8, 8)))
    return config


def get_t16_config():
    """Returns the ViT-T configuration."""
    config = get_base_config()
    config.update(dict(
        hidden_size=192,
        intermediate_size=768,
        num_attention_heads=3,
        representation_size=192,
    ))
    return config


def get_t32_config():
    """Returns the ViT-T/32 configuration."""
    config = get_t16_config()
    config.update(dict(patch_size=(32, 32)))
    return config


def get_t8_config():
    """Returns the ViT-T/8 configuration."""
    config = get_t16_config()
    config.update(dict(patch_size=(8, 8)))
    return config


def get_t4_config():
    """Returns the ViT-T/4 configuration."""
    config = get_t16_config()
    config.update(dict(patch_size=(4, 4)))
    return config


def get_m16_config():
    """Returns the ViT-m (u/mu/micro) configuration."""
    hidden_size = 96
    config = get_base_config()
    config.update(dict(
        hidden_size=hidden_size,
        intermediate_size=hidden_size * 4,
        num_attention_heads=3,
    ))
    return config


def get_n16_config():
    """Returns the ViT-n (nano) configuration."""
    hidden_size = 48
    config = get_base_config()
    config.update(dict(
        hidden_size=hidden_size,
        intermediate_size=hidden_size * 4,
        num_attention_heads=3,
    ))
    return config


CONFIGS = {
    'vit_n16': {
        'config': get_n16_config(),
    },
    'vit_m16': {
        'config': get_m16_config(),
    },
    'vit_t4': {
        'config': get_t4_config(),
    },
    'vit_t8': {
        'config': get_t8_config(),
    },
    'vit_t16': {
        'config': get_t16_config(),
    },
    'vit_t32': {
        'config': get_t32_config(),
    },
    'vit_s8': {
        'config': get_s8_config(),
    },
    'vit_s16': {
        'config': get_s16_config(),
    },
    'vit_s32': {
        'config': get_s32_config(),
    },
    'vit_b8': {
        'config': get_b8_config(),
    },
    'vit_b16': {
        'config': get_b16_config(),
    },
    'vit_b32': {
        'config': get_b32_config(),
    },
    'vit_l16': {
        'config': get_l16_config(),
    },
    'vit_l32': {
        'config': get_l32_config(),
    },
    'vit_h14': {
        'config': get_h14_config(),
    }
}
