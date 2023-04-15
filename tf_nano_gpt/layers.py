from typing import Optional, Tuple, Union

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
    def __init__(
        self, nx, config, top_k: int = 16, reduction: int = 8, *args, **kwargs
    ):
        super().__init__(nx=nx, config=config, *args, **kwargs)
        self.top_k = top_k
        self.reduction = reduction

    @classmethod
    def apply_casual_attention_mask(cls, w):
        _, _, nd, ns = shape_list(w)
        b = cls.causal_attention_mask(nd, ns, dtype=w.dtype)
        b = tf.reshape(b, [1, 1, nd, ns])
        return w * b - 1e4 * (1 - b)

    def call(self, inputs, *args, **kwargs):
        self._last_inputs = inputs
        # print(f"calling approx attention on inputs: {inputs.shape}")
        outputs = super().call(inputs, *args, **kwargs)
        self._last_outputs = outputs
        return outputs

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
        bs, heads, nd, ns = shape_list(q)

        # q_reduced = tf.reshape(q, [bs, heads, nd, ns // self.reduction, self.reduction])
        # k_reduced = tf.reshape(k, [bs, heads, nd, ns // self.reduction, self.reduction])
        #
        # q_reduced = tf.reduce_sum(q_reduced, axis=-1)
        # k_reduced = tf.reduce_sum(k_reduced, axis=-1)
        #
        # w = tf.matmul(q_reduced, k_reduced, transpose_b=True)
        w = tf.matmul(q, k, transpose_b=True)
        scale = 1.0
        if self.scale:
            dk = tf.cast(shape_list(k)[-1], dtype=w.dtype)
            scale = tf.math.sqrt(dk)  # scale attention_scores
            w = w / scale

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

        # Mask heads if we want to
        if head_mask is not None:
            w = w * head_mask

        logits, indices = tf.math.top_k(w, k=self.top_k, sorted=True)
        logits_dims = shape_list(logits)[:-1]
        sampled_indices = tf.random.categorical(
            tf.reshape(logits, [-1, self.top_k]), num_samples=1
        )
        sampled_indices = tf.reshape(sampled_indices, logits_dims)
        sampled_indices = tf.stop_gradient(sampled_indices)

        # exact scores
        k_sampled = tf.gather(k, sampled_indices, batch_dims=2)
        w_top_logits = tf.reduce_sum(q * k_sampled, axis=-1, keepdims=True) / scale

        max_logits = tf.reduce_max(logits, axis=-1, keepdims=True)
        denom = tf.reduce_sum(tf.exp(logits - max_logits), axis=-1, keepdims=True)
        probs = tf.exp(w_top_logits - max_logits) / denom

        v_sampled = tf.gather(v, sampled_indices, batch_dims=2)
        attention_result = probs * v_sampled
        attention_result = self.attn_dropout(attention_result, training=training)

        outputs = [attention_result]
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


class TopKFrozenEmbeddings(layers.Layer):
    def set_embeddings_matrix(self, embeddings: tf.Tensor, dtype: tf.DType = None):
        self.r = 8
        self.top_k = 5
        if dtype is not None:
            embeddings = tf.cast(embeddings, dtype=dtype)
        self.embeddings = embeddings
        self.embeddings_reduced = self.reduce_dim(embeddings)

    @property
    def hidden_size(self) -> int:
        return self.embeddings.shape[-1]

    @property
    def vocab_size(self) -> int:
        return self.embeddings.shape[0]

    def reduce_dim(self, x: tf.Tensor) -> tf.Tensor:
        r = self.r
        d = x.shape[-1]
        return tf.reduce_sum(tf.reshape(x, [-1, d // r, r]), -1)

    def call(self, inputs: tf.Tensor, mode: str = "embedding", *args, **kwargs):
        if mode == "embedding":
            return self._embedding(inputs)
        elif mode == "linear":
            return self._linear(inputs)
        elif mode == "test":
            return self._linear_exact(inputs)
        else:
            raise ValueError(f"mode {mode} is not valid.")

    def _embedding(self, input_ids):
        """Applies embedding based on inputs tensor."""
        return tf.gather(self.embeddings, input_ids)

    def _linear_exact(self, inputs):
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

    def _linear(self, inputs: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        dims = shape_list(inputs)
        x = tf.reshape(inputs, [-1, self.hidden_size])
        # [batch, dim]
        x_reduced = self.reduce_dim(x)
        # [batch, dim // r]

        approx_scores = tf.matmul(x_reduced, self.embeddings_reduced, transpose_b=True)
        # [batch, num_embeddings]

        _, top_indices = tf.math.top_k(approx_scores, k=self.top_k)

        # computing exact scores on sampled top approx scores
        top_embeddings = tf.gather(self.embeddings, top_indices, batch_dims=0)
        logits = tf.reduce_sum(tf.expand_dims(x, 1) * top_embeddings, -1)
        scores = tf.nn.softmax(logits)

        probs = tf.reduce_max(scores, axis=-1)
        # [batch, ]
        indices = tf.gather(top_indices, tf.argmax(scores, axis=-1), batch_dims=1)
        # [batch, ]
        probs = tf.reshape(probs, dims[:-1])
        indices = tf.reshape(indices, dims[:-1])
        return probs, indices

    @classmethod
    def get_loss(
        cls, y_pred_probs: tf.Tensor, y_pred_indices: tf.Tensor, y_true: tf.Tensor
    ) -> tf.Tensor:
        y_true_labels = y_true[..., 0]
        y_true_mask = y_true[..., 1]

        y_true_mask = tf.cast(y_true_mask, y_pred_probs.dtype)

        y_true = tf.cast(y_pred_indices == y_true_labels, y_pred_probs.dtype)
        y_pred_probs = tf.expand_dims(y_pred_probs, -1)
        y_true = tf.expand_dims(y_true, -1)

        losses = tf.losses.binary_crossentropy(y_true, y_pred_probs)

        masked_loss = losses * y_true_mask
        return tf.reduce_mean(tf.reduce_mean(masked_loss, axis=-1))

    @classmethod
    def get_simple_accuracy_metric(cls, y_true, y_pred):
        y_true_flat = tf.reshape(y_true[..., 0], [-1])
        y_pred_flat = tf.reshape(y_pred, [-1])
        indices = tf.where(tf.reshape(y_true[..., 1], [-1]) > 0)

        equals = tf.gather(y_true_flat, indices) == tf.gather(y_pred_flat, indices)
        equals = tf.cast(equals, dtype=tf.float32)
        return tf.reduce_mean(tf.reduce_mean(equals, axis=-1))

    @classmethod
    def get_named_loss_layer(cls, name: str):
        return tf.keras.layers.Lambda(lambda x: cls.get_loss(*x), name=name)

    @classmethod
    def get_named_accuracy_layer(cls, name: str):
        return tf.keras.layers.Lambda(
            lambda x: cls.get_simple_accuracy_metric(*x), name=name
        )
