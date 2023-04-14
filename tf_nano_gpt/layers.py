from typing import Optional, Union

import tensorflow as tf
from transformers.modeling_tf_utils import TFConv1D, get_initializer
from transformers.models.gpt2.modeling_tf_gpt2 import TFAttention
from transformers.tf_utils import shape_list, stable_softmax

layers = tf.keras.layers


class FactorizedConv1D(layers.Layer):
    def __init__(self, nf: int, top_k: int, initializer_range: float = 0.02, **kwargs):
        super().__init__(**kwargs)
        self.nf = nf
        self.top_k = top_k
        self.initializer_range = initializer_range

        self.weight = self.add_weight(
            "weight",
            shape=[self.top_k, self.top_k],
            initializer=get_initializer(self.initializer_range),
        )
        self.bias = self.add_weight(
            "bias", shape=[1, self.nf], initializer=tf.zeros_initializer()
        )

    @classmethod
    def from_huggingface_conv1d(cls, hf_layer, top_k: int):
        layer = cls(
            hf_layer.nf,
            top_k,
            initializer_range=hf_layer.initializer_range,
            name=f"{hf_layer.name}-f{top_k}",
        )
        layer.init_from_conv1d(hf_layer)
        return layer

    def init_from_conv1d(self, layer: layers.Conv1D):
        w, b = layer.get_weights()
        if len(w.shape) == 3:
            # remove first axis
            w = w.reshape((w.shape[1], w.shape[2]))

        if len(b.shape) == 1:
            b = b.reshape((1, self.nf))

        s, u, v = tf.linalg.svd(w)

        s = tf.linalg.diag(s[: self.top_k])
        self.u = u[:, : self.top_k]
        self.v = v[:, : self.top_k]

        self.set_weights([s.numpy(), b])

        w_approx = tf.matmul(self.u, s)
        w_approx = tf.matmul(w_approx, self.v, transpose_b=True)
        total_error = tf.reduce_mean(tf.abs(w)).numpy()
        error = tf.reduce_mean(tf.abs(w_approx - w)).numpy() / total_error

        print(
            f"SVD relative compression error for {type(layer)} with name '{layer.name}' "
            f"is {error:.3f} w={w.shape} bias={b.shape}"
        )

        if tf.keras.mixed_precision.global_policy().name == "mixed_float16":
            self.u = tf.cast(self.u, tf.float16)
            self.v = tf.cast(self.v, tf.float16)

    def call(self, x, **kwargs):
        bz, sl, channels = shape_list(x)

        x = tf.reshape(x, [-1, channels])
        x = tf.matmul(x, self.u)
        x = tf.matmul(x, self.weight)
        x = tf.matmul(x, self.v, transpose_b=True) + self.bias

        x = tf.reshape(x, [bz, sl, self.nf])

        return x


class SVDFactorizationModel(tf.keras.Model):
    def __init__(self, w, top_k: int, **kwargs):
        super().__init__(**kwargs)

        s, u, v = tf.linalg.svd(w)

        if tf.keras.mixed_precision.global_policy().name == "mixed_float16":
            u = tf.cast(u, tf.float16)
            v = tf.cast(v, tf.float16)

        alpha = s[0] / 10

        s = tf.linalg.diag(s[:top_k])
        noise = tf.random.uniform(
            shape=s.shape, dtype=s.dtype, minval=-alpha, maxval=alpha
        )
        s = s + noise

        self.w = w
        self.u = u[:, :top_k]
        self.v = v[:, :top_k]
        self.s = self.add_weight(
            "s",
            shape=s.numpy().shape,
            initializer=lambda *args, **_kwargs: s,
        )

    def call(self, inputs, training=None, mask=None):
        if tf.keras.mixed_precision.global_policy().name == "mixed_float16":
            inputs = tf.cast(self.s, tf.float16)
        else:
            inputs = self.s

        w_approx = tf.matmul(self.u, inputs)
        w_approx = tf.matmul(w_approx, self.v, transpose_b=True)
        total_error = tf.reduce_mean(tf.square(self.w - w_approx))
        self.add_loss(1000 * total_error)
        return total_error


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


class ApproxTFAttention(TFAttention):
    def __init__(self, nx, config, num_casual_blocks: int = 1, *args, **kwargs):
        super().__init__(nx=nx, config=config, *args, **kwargs)
        self.num_casual_blocks = num_casual_blocks

    @classmethod
    def apply_casual_attention_mask(cls, w):
        _, _, nd, ns = shape_list(w)
        b = cls.causal_attention_mask(nd, ns, dtype=w.dtype)
        b = tf.reshape(b, [1, 1, nd, ns])
        return w * b - 1e4 * (1 - b)

    def _attn(
        self,
        q: tf.Tensor,
        k: tf.Tensor,
        v: tf.Tensor,
        attention_mask: Optional[tf.Tensor] = None,
        head_mask: Optional[tf.Tensor] = None,
        output_attentions: bool = False,
        training: bool = False,
    ):
        # q, k, v have shape [batch, heads, sequence, features]
        # is now [batch, heads, dst_sequence, src_sequence]
        q = (q[..., ::2] + q[..., 1::2]) / 2
        k = (k[..., ::2] + k[..., 1::2]) / 2
        w = tf.matmul(q, k, transpose_b=True)
        if self.scale:
            dk = tf.cast(shape_list(k)[-1], dtype=w.dtype)  # scale attention_scores
            w = w / tf.math.sqrt(dk)

        if not self.is_cross_attention:
            # if only "normal" attention layer implements causal mask

            # w has shape [batch, heads, dst_sequence, src_sequence], where information flows from src to dst.
            _, _, nd, ns = shape_list(w)
            b = self.causal_attention_mask(nd, ns, dtype=w.dtype)
            b = tf.reshape(b, [1, 1, nd, ns])
            w = w * b - 1e4 * (1 - b)

        if attention_mask is not None:
            # Apply the attention mask
            attention_mask = tf.cast(attention_mask, dtype=w.dtype)
            w = w + attention_mask

        w = casual_softmax(w, self.num_casual_blocks)
        w = self.attn_dropout(w, training=training)

        # Mask heads if we want to
        if head_mask is not None:
            w = w * head_mask

        outputs = [tf.matmul(w, v)]
        if output_attentions:
            outputs.append(w)
        return outputs


def casual_softmax(x: tf.Tensor, num_blocks: int = 2) -> tf.Tensor:
    if num_blocks == 1:
        return stable_softmax(x, axis=-1)

    bs, heads, sl, sl = shape_list(x)

    x_splits = tf.split(x, num_blocks, axis=2)
    x_probs = []
    for b, x_split in enumerate(x_splits):
        split_size = sl // num_blocks
        zeros_mask = tf.zeros_like(x_split[..., (b + 1) * split_size :])
        x_split = x_split[..., : (b + 1) * split_size]
        x_split = stable_softmax(x_split, axis=-1)
        x_split = tf.concat([x_split, zeros_mask], axis=-1)
        x_probs.append(x_split)

    return tf.concat(x_probs, axis=2)
