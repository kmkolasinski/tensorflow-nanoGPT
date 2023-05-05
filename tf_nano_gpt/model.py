from typing import Optional, Tuple

import keras_nlp
import tensorflow as tf
from transformers import TFGPT2Model

from tf_nano_gpt.layers import FrozenEmbeddings, LoRADense


class GPT2Tokenizer(keras_nlp.tokenizers.BytePairTokenizer):
    def __init__(
        self,
        vocabulary: str = "tokenizer/vocab.json",
        merges: str = "tokenizer/merges.txt",
        pad_token_str: str = "::",
        start_token_str: str = "???",
        stop_token_str: str = "<|endoftext|>",
        **kwargs
    ):
        super().__init__(vocabulary, merges, **kwargs)
        self.pad_token_str = pad_token_str
        self.start_token_str = start_token_str
        self.stop_token_str = stop_token_str

    @property
    def pad_token_id(self) -> int:
        return self.token_to_id(self.pad_token_str)

    @property
    def stop_token_id(self) -> int:
        return self.token_to_id(self.stop_token_str)

    @property
    def start_token_id(self) -> int:
        return self.token_to_id(self.start_token_str)

    def pad_or_slice(self, tokens, seq_length: int, pad_value: Optional[int] = None):
        if pad_value is None:
            pad_value = self.pad_token_id

        tokens = tokens[:seq_length]
        tokens = tf.pad(
            tokens,
            paddings=[[0, seq_length - tf.shape(tokens)[0]]],
            constant_values=pad_value,
        )
        tokens.set_shape((seq_length,))
        return tokens

    @tf.function(
        input_signature=(
            tf.TensorSpec((), tf.string),
            tf.TensorSpec((), tf.string),
        )
    )
    def tokenize_sample(
        self, context_text: tf.Tensor, target_text: tf.Tensor
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        tokens = self([context_text, target_text])
        context_tokens, target_tokens = tokens[0], tokens[1]
        target_tokens = tf.concat(
            [[self.start_token_id], target_tokens, [self.stop_token_id]], 0
        )
        return context_tokens, target_tokens


def freeze_embeddings(base_model: TFGPT2Model, dtype: tf.DType = tf.float16):
    wte_layer = FrozenEmbeddings(base_model.transformer.wte, dtype, trainable=False)
    base_model.transformer.wte = wte_layer
    wpe = base_model.transformer.wpe.numpy()
    base_model.transformer.wpe = tf.cast(wpe, dtype)


def freeze_layers(
    base_model: TFGPT2Model,
    num_blocks_to_freeze: int = 8,
    use_lora: bool = False,
    lora_rank: int = 16,
):
    for block in base_model.transformer.h[:num_blocks_to_freeze]:
        block.trainable = False

    if use_lora:
        for block in base_model.transformer.h[num_blocks_to_freeze:]:
            block.mlp.trainable = False
            block.attn.c_attn = LoRADense(block.attn.c_attn, r=lora_rank)
            block.attn.c_proj = LoRADense(block.attn.c_proj, r=lora_rank)
