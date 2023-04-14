import math
from dataclasses import dataclass

import tensorflow as tf

layers = tf.keras.layers


class MLP(tf.keras.Model):
    def __init__(self, n_embd: int, bias: bool = True, dropout: float = 0.0, **kwargs):
        super().__init__(**kwargs)
        self.c_fc = layers.Dense(4 * n_embd, use_bias=bias)
        self.c_proj = layers.Dense(n_embd, use_bias=bias)
        self.dropout = layers.Dropout(dropout)

    def call(self, inputs: tf.Tensor, *args, **kwargs) -> tf.Tensor:
        x = self.c_fc(inputs)
        x = tf.nn.gelu(x, approximate=True)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

    def init_weights(self, n_layer: int):
        # apply special scaled init to the residual projections, per GPT-2 paper
        weights = self.c_proj.get_weights()
        kernel = tf.random.normal(
            weights[0].shape, mean=0.0, stddev=0.02 / math.sqrt(2 * n_layer)
        )
        self.c_proj.set_weights([kernel] + weights[1:])


class CasualSelfAttentionBlock(tf.keras.Model):
    def __init__(
        self,
        n_embd: int,
        n_head: int,
        bias: bool = True,
        dropout: float = 0.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.ln_1 = layers.LayerNormalization()
        self.attn = layers.MultiHeadAttention(
            num_heads=n_head, key_dim=n_embd // n_head, dropout=dropout, use_bias=bias
        )
        self.ln_2 = layers.LayerNormalization()
        self.mlp = MLP(n_embd=n_embd, bias=bias, dropout=dropout, name="mlp")

    def call(self, inputs: tf.Tensor, *args, **kwargs) -> tf.Tensor:
        x = self.ln_1(inputs)
        x = x + self.attn(query=x, value=x, use_causal_mask=True)
        x = x + self.mlp(self.ln_2(x))
        return x


@dataclass
class GPTConfig:
    sequence_length: int = 1024
    vocab_size: int = 50304
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True


class TokenAndPositionEmbedding(layers.Layer):
    def __init__(
        self,
        sequence_length: int,
        vocab_size: int,
        n_embd: int,
        dropout: float = 0.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.token_emb = layers.Embedding(input_dim=vocab_size, output_dim=n_embd)
        self.pos_emb = layers.Embedding(input_dim=sequence_length, output_dim=n_embd)
        self.dropout = layers.Dropout(dropout)

    def call(self, x: tf.Tensor, **kwargs) -> tf.Tensor:
        maxlen = tf.shape(x)[-1]
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        return self.dropout(x + positions)


class GPT(tf.keras.Model):
    def __init__(self, config: GPTConfig, **kwargs):
        super().__init__(**kwargs)
        assert config.vocab_size is not None  # nosec
        assert config.sequence_length is not None  # nosec
        self.config = config

        self.embeddings = TokenAndPositionEmbedding(
            sequence_length=config.sequence_length,
            vocab_size=config.vocab_size,
            n_embd=config.n_embd,
            dropout=config.dropout,
        )

        self.blocks = [
            CasualSelfAttentionBlock(
                n_head=config.n_head,
                n_embd=config.n_embd,
                bias=config.bias,
                dropout=config.dropout,
                name=f"block_{i}",
            )
            for i in range(config.n_layer)
        ]
        self.ln_f = layers.LayerNormalization()
        self.lm_head = layers.Dense(config.vocab_size, use_bias=False)

    def build(self, input_shape):
        super().build(input_shape)
        for block in self.blocks:
            # apply special scaled init to the residual projections, per GPT-2 paper
            block.mlp.init_weights(self.config.n_layer)

    def call(self, inputs: tf.Tensor, **kwargs) -> tf.Tensor:
        x = self.embeddings(inputs)
        self._post_embeddings = x
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        return logits
