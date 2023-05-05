from typing import Union

import tensorflow as tf
from transformers.modeling_tf_utils import TFConv1D, TFSharedEmbeddings
from transformers.tf_utils import shape_list

layers = tf.keras.layers


class LoRADense(layers.Layer):
    def __init__(
        self,
        base: Union[layers.Dense, layers.Conv1D, TFConv1D],
        r: int = 8,
        lora_alpha: float = 1.0,
        stop_gradients: bool = False,
        use_bias: bool = False,
        **kwargs,
    ):
        super().__init__(name=f"{base.name}-lora-{r}", **kwargs)

        self.use_bias = use_bias
        self.r = r
        self.scale = lora_alpha / self.r

        if isinstance(base, (layers.Dense, layers.Conv1D)):
            units = base.units
        else:
            units = base.nf

        self.base = base
        self.base.trainable = False
        self.stop_gradients = stop_gradients

        self.lora_down = layers.Dense(
            self.r,
            use_bias=self.use_bias,
            kernel_initializer=tf.keras.initializers.RandomNormal(stddev=1 / self.r),
        )
        self.lora_up = layers.Dense(
            units,
            use_bias=self.use_bias,
            kernel_initializer=tf.keras.initializers.Zeros(),
        )

    def call(self, inputs, **kwargs):
        x0 = self.base(inputs)
        if self.stop_gradients:
            x0 = tf.stop_gradient(x0)
        x1 = self.lora_up(self.lora_down(inputs)) * self.scale
        return x0 + x1


class FrozenEmbeddings(layers.Layer):
    def __init__(
        self, base_layer: TFSharedEmbeddings, dtype: tf.DType = None, *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        embeddings = base_layer.weights[0].numpy()
        if dtype is not None:
            embeddings = tf.cast(embeddings, dtype=dtype)
        self.embeddings = embeddings

    @property
    def hidden_size(self) -> int:
        return self.embeddings.shape[-1]

    @property
    def vocab_size(self) -> int:
        return self.embeddings.shape[0]

    def call(self, inputs: tf.Tensor, mode: str = "embedding", *args, **kwargs):
        if mode == "embedding":
            return self._embedding(inputs)
        elif mode == "linear":
            return self._linear(inputs)
        else:
            raise ValueError(f"mode {mode} is not valid.")

    def _embedding(self, input_ids: tf.Tensor) -> tf.Tensor:
        """Applies embedding based on inputs tensor."""

        return tf.gather(self.embeddings, input_ids)

    def _linear(self, inputs):
        """
        Computes logits by running inputs through a linear layer.

        Args:
            inputs: A float32 tensor with shape [..., hidden_size]

        Returns:
            float32 tensor with shape [..., vocab_size].
        """
        dims = shape_list(inputs)
        x = tf.reshape(inputs, [-1, self.hidden_size])
        logits = tf.matmul(x, self.embeddings, transpose_b=True)
        return tf.reshape(logits, dims[:-1] + [self.vocab_size])
