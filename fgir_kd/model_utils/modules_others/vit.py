import torch
from torch import nn
from einops import repeat, rearrange
from einops.layers.torch import Reduce

from .transformer import Transformer
from .download_load_pt_weights import load_pretrained_weights


class LearnedPositionalEmbedding1D(nn.Module):
    """Adds (optionally learned) positional embeddings to the inputs."""

    def __init__(self, seq_len, dim):
        super().__init__()
        self.pos_embedding = nn.Parameter(torch.zeros(1, seq_len, dim))

    def forward(self, x):
        """Input has shape `(batch_size, seq_len, emb_dim)`"""
        x = x + self.pos_embedding

        return x


class ViT(nn.Module):
    def __init__(self, config, pretrained=False):
        super().__init__()

        # Patch embedding
        self.patch_embedding = nn.Conv2d(
            in_channels=config.num_channels, out_channels=config.hidden_size,
            kernel_size=(config.fh, config.fw), stride=(config.patch_stride, config.patch_stride))

        # Class token
        if 'cls' in config.classifier:
            self.class_token = nn.Parameter(torch.randn(1, 1, config.hidden_size))

        # Positional embedding
        if config.pos_embedding_type == 'learned':
            self.positional_embedding = LearnedPositionalEmbedding1D(config.seq_len, config.hidden_size)

        # Transformer encoder
        self.encoder = Transformer(
            num_layers=config.num_hidden_layers,
            dim=config.hidden_size,
            num_heads=config.num_attention_heads,
            ff_dim=config.intermediate_size,
            hidden_dropout_prob=config.hidden_dropout_prob,
            attention_probs_dropout_prob=config.attention_probs_dropout_prob,
            layer_norm_eps=config.layer_norm_eps,
            sd=config.sd,
        )

        if config.encoder_norm:
            self.encoder_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        # Initialize weights
        self.init_weights()

        if pretrained:
            load_pretrained_weights(self, config, config.model_name)

    @torch.no_grad()
    def init_weights(self):
        def _init(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.normal_(m.bias, std=1e-6)
        self.apply(_init)
        if hasattr(self, 'positional_embedding'):
            nn.init.normal_(self.positional_embedding.pos_embedding, std=0.02)
        if hasattr(self, 'class_token'):
            nn.init.constant_(self.class_token, 0)

    def forward(self, images):
        """
        x (tensor): b k c fh fw -> b s d
        """
        x = self.patch_embedding(images)
        x = rearrange(x, 'b d gh gw -> b (gh gw) d')
        b, _, _ = x.shape

        if hasattr(self, 'class_token'):
            cls_tokens = repeat(self.class_token, '1 1 d -> b 1 d', b=b)
            x = torch.cat((cls_tokens, x), dim=1)  # b s+1 d

        if hasattr(self, 'positional_embedding'):
            x = self.positional_embedding(x)

        x = self.encoder(x)

        if hasattr(self, 'encoder_norm'):
            x = self.encoder_norm(x)

        return x
