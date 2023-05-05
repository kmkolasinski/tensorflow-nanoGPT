from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import numpy as np
import tensorflow as tf
from transformers import GPT2Config, TFGPT2Model
from transformers.activations_tf import get_tf_activation
from transformers.modeling_tf_outputs import TFBaseModelOutputWithPastAndCrossAttentions
from transformers.modeling_tf_utils import (
    TFConv1D,
    TFModelInputType,
    TFSharedEmbeddings,
    get_initializer,
    unpack_inputs,
)
from transformers.models.gpt2.modeling_tf_gpt2 import (
    TFMLP,
    TFAttention,
    TFBlock,
    TFGPT2PreTrainedModel,
)
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


class ContextTFAttention(TFAttention):
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

    def init(self, base: TFAttention):
        args = dict(
            layer_past=None,
            attention_mask=None,
            head_mask=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            use_cache=False,
            output_attentions=False,
            training=False,
        )
        shape = (1, 8, self.c_proj.nx)
        x = tf.random.uniform(shape, dtype=tf.float16)
        self(x, **args)
        base(x, **args)
        self.set_weights(base.get_weights())

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
        assert not self.is_cross_attention
        assert self.scale
        bs, heads, sl, nf = shape_list(q)

        w0 = tf.matmul(q[..., : sl // 2, :], k[..., : sl // 2, :], transpose_b=True)
        w1 = tf.matmul(q[..., sl // 2 :, :], k[..., :, :], transpose_b=True)

        dk = tf.cast(shape_list(k)[-1], dtype=k.dtype)  # scale attention_scores
        w0 = w0 / tf.math.sqrt(dk)
        w1 = w1 / tf.math.sqrt(dk)

        w0 = tf.pad(w0, [(0, 0), (0, 0), (0, 0), (0, sl // 2)])
        wp = tf.concat([w0, w1], axis=-2)
        w = self.apply_casual_attention_mask(wp)

        if attention_mask is not None:
            # Apply the attention mask
            attention_mask = tf.cast(attention_mask, dtype=w.dtype)
            w = w + attention_mask

        w = stable_softmax(w, axis=-1)
        w = self.attn_dropout(w, training=training)

        # Mask heads if we want to
        if head_mask is not None:
            w = w * head_mask

        outputs = [tf.matmul(w, v)]
        if output_attentions:
            outputs.append(w)
        return outputs


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
        self.r = 4
        self.top_k = 10
        self.num_samples = 64
        if dtype is not None:
            embeddings = tf.cast(embeddings, dtype=dtype)
        self.embeddings = embeddings
        self.embeddings_reduced = self.reduce_dim(embeddings)

    def set_r(self, r):
        self.r = r
        self.embeddings_reduced = self.reduce_dim(self.embeddings)

    def set_inference_constraint_extra_tokes_ids(
        self, tokenizer, tokens_ids: List[int]
    ):
        self._keras_tokenizer = tokenizer
        self.inference_constraint_extra_tokes_ids = tokens_ids

    @property
    def hidden_size(self) -> int:
        return self.embeddings.shape[-1]

    @property
    def vocab_size(self) -> int:
        return self.embeddings.shape[0]

    def reduce_dim(self, x: tf.Tensor) -> tf.Tensor:
        r = self.r
        d = x.shape[-1]
        scale = tf.sqrt(tf.cast(r, x.dtype))
        # return scale * tf.reduce_mean(tf.reshape(x, [-1, d // r, r]), -1)
        return scale * x[..., ::r]

    def call(
        self,
        inputs: tf.Tensor,
        targets_ids: tf.Tensor = None,
        output_tokens_ids: tf.Tensor = None,
        mode: str = "embedding",
        training: bool = True,
        *args,
        **kwargs,
    ):
        # print(f"is training mode enabled: {training}")
        if mode == "embedding":
            return self._embedding(inputs)
        elif mode == "linear_v1":
            self._last_inputs = (inputs, targets_ids)
            return self._linear_v1(inputs, targets_ids)
        elif mode == "linear_v2":
            self._last_inputs = (inputs, targets_ids)
            return self._linear_v2(inputs, targets_ids)
        elif mode == "linear_sampled":
            self._last_inputs = (inputs, targets_ids)
            return self._linear_sampled(inputs, targets_ids, training=training)
        elif mode == "test":
            return self._linear_exact(inputs)
        elif mode == "constrained_test":
            return self._linear_exact_with_constraint(inputs, output_tokens_ids)
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

    @classmethod
    def inputs_ids_to_constraint_ids(cls, tokenizer, inputs_ids, extra_ids):
        decoded_tokens = tokenizer.detokenize(tf.reshape(inputs_ids, [-1, 1]))
        decoded_tokens = tf.unique(tf.strings.strip(decoded_tokens))[0]
        # possible_tokens = tf.concat(
        #     [
        #         decoded_tokens,
        #         decoded_tokens + " ",
        #         " " + decoded_tokens,
        #     ],
        #     axis=0,
        # )
        unique_inputs_ids = tf.unique(tokenizer.tokenize(decoded_tokens).flat_values)
        unique_inputs_ids = unique_inputs_ids[0]
        extra_tokens_ids = tf.constant(extra_ids)
        unique_ids = tf.concat([unique_inputs_ids, extra_tokens_ids], axis=0)
        unique_ids = tf.unique(unique_ids)[0]
        return unique_ids

    def _linear_exact_with_constraint(self, inputs, output_tokens_ids):
        dims = shape_list(inputs)
        x = tf.reshape(inputs, [-1, self.hidden_size])

        # embeddings is no [num_unique_tokens, 768]
        embeddings = tf.gather(self.embeddings, output_tokens_ids)
        logits = tf.matmul(x, embeddings, transpose_b=True)

        tokens_ids_mapping = tf.gather(tf.range(self.vocab_size), output_tokens_ids)
        logits = tf.reshape(logits, dims[:-1] + [tf.shape(embeddings)[0]])
        return logits, tokens_ids_mapping

    def _linear_v1(
        self, inputs: tf.Tensor, targets_ids: tf.Tensor
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        return self._linear_exact(inputs)

    def _linear_v2(self, inputs: tf.Tensor, targets_ids: tf.Tensor):
        dims = shape_list(inputs)
        x = tf.reshape(inputs, [-1, self.hidden_size])
        logits = tf.matmul(x, self.embeddings, transpose_b=True)
        scores = tf.nn.softmax(logits)

        targets_probs = tf.gather(scores, tf.reshape(targets_ids, [-1]), batch_dims=1)
        predicted_ids = tf.argmax(scores, -1)

        targets_probs = tf.reshape(targets_probs, dims[:-1])
        predicted_ids = tf.reshape(predicted_ids, dims[:-1])

        probs = tf.reshape(scores, dims[:-1] + [self.vocab_size])
        logits = tf.reshape(logits, dims[:-1] + [self.vocab_size])
        return probs, logits, targets_probs, predicted_ids

    def _linear_sampled(
        self, inputs: tf.Tensor, targets_ids: tf.Tensor, training: bool = True
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        x = tf.reshape(inputs, [-1, self.hidden_size])
        # [batch, dim]
        targets_ids = tf.reshape(targets_ids, [-1, 2])
        # [batch]
        indices = tf.where(targets_ids[..., 1] > 0)[:, 0]

        if training:
            sampled_indices = tf.random.uniform(
                (self.num_samples,),
                minval=0,
                maxval=tf.shape(indices)[0],
                dtype=tf.int32,
            )
            indices = tf.gather(indices, sampled_indices)

        # print(f"indices: {indices}")
        targets_ids = tf.gather(targets_ids[..., 0], indices)
        x = tf.gather(x, indices)

        logits = tf.matmul(x, self.embeddings, transpose_b=True)
        scores = tf.nn.softmax(logits)
        targets_probs = tf.gather(scores, targets_ids, batch_dims=1)
        predicted_ids = tf.argmax(scores, -1)

        return targets_probs, targets_ids, predicted_ids

    @classmethod
    def get_losses(cls, y_true: tf.Tensor, predictions, mode: str):
        if mode == "linear_v1":
            loss = custom_masked_loss_fn(y_true, predictions)
            y_pred = tf.argmax(predictions, axis=-1)
            accuracy = custom_accuracy(y_true, y_pred)
            return loss, accuracy
        elif mode == "linear_v2":
            probs, logits, target_probs, predicted_ids = predictions
            loss = custom_masked_loss_from_probs(y_true, target_probs)
            accuracy = custom_accuracy(y_true, predicted_ids)
            return loss, accuracy
        elif mode == "linear_sampled":
            targets_probs, targets_ids, predicted_ids = predictions
            loss = custom_sampled_loss_from_probs(targets_probs)
            accuracy = custom_sampled_accuracy(targets_ids, predicted_ids)
            return loss, accuracy
        else:
            raise NotImplementedError

    def _linear(self, inputs: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        dims = shape_list(inputs)
        x = tf.reshape(inputs, [-1, self.hidden_size])
        # [batch, dim]
        x_reduced = self.reduce_dim(x)
        # [batch, dim // r]

        approx_scores = tf.matmul(x_reduced, self.embeddings_reduced, transpose_b=True)
        # [batch, num_embeddings]
        approx_scores = tf.nn.softmax(approx_scores)
        top_scores, top_indices = tf.math.top_k(approx_scores, k=self.top_k)

        # computing exact scores on sampled top approx scores
        top_embeddings = tf.gather(self.embeddings, top_indices, batch_dims=0)
        logits = tf.reduce_sum(tf.expand_dims(x, 1) * top_embeddings, -1)

        scores = (tf.nn.softmax(logits) + top_scores) / 2

        probs = tf.reduce_max(scores, axis=-1)
        # [batch, ]
        indices = tf.gather(top_indices, tf.argmax(scores, axis=-1), batch_dims=1)
        # [batch, ]
        probs = tf.reshape(probs, dims[:-1])
        indices = tf.reshape(indices, dims[:-1])
        return probs

    @classmethod
    def get_loss(cls, y_pred_probs: tf.Tensor, y_true: tf.Tensor) -> tf.Tensor:
        y_true_labels = y_true[..., 0]
        y_true_mask = y_true[..., 1]

        losses = -tf.math.log(tf.clip_by_value(y_pred_probs, 1e-7, 1.0))
        y_true_mask = tf.cast(y_true_mask, y_pred_probs.dtype)
        masked_loss = losses * y_true_mask
        return tf.reduce_mean(tf.reduce_mean(masked_loss, axis=-1))

    @classmethod
    def get_simple_accuracy_metric(cls, y_true: tf.Tensor, y_pred: tf.Tensor):
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


def custom_masked_loss_from_probs(y_true, y_pred_probs):
    y_true_labels = y_true[..., 0]
    y_true_mask = y_true[..., 1]

    losses = -tf.math.log(tf.clip_by_value(y_pred_probs, 0.0001, 0.9999))
    y_true_mask = tf.cast(y_true_mask, y_pred_probs.dtype)
    masked_loss = losses * y_true_mask
    return tf.reduce_mean(tf.reduce_mean(masked_loss, axis=-1))


def custom_sampled_loss_from_probs(y_pred_probs):
    losses = -tf.math.log(tf.clip_by_value(y_pred_probs, 0.0001, 0.9999))
    return tf.reduce_mean(losses)


def custom_masked_loss_fn(y_true, y_pred, from_logits: bool = True):
    y_true_labels = y_true[..., 0]
    y_true_mask = y_true[..., 1]
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=from_logits, reduction="none"
    )
    masked_loss = loss_fn(y_true_labels, y_pred) * tf.cast(y_true_mask, y_pred.dtype)
    return tf.reduce_mean(tf.reduce_mean(masked_loss, axis=-1))


def custom_sampled_accuracy(y_true, y_pred):
    equals = y_true == tf.cast(y_pred, y_true.dtype)
    equals = tf.cast(equals, tf.float32)
    return tf.reduce_mean(tf.reduce_mean(equals, axis=-1))


def custom_accuracy(y_true, y_pred):
    y_true_flat = tf.reshape(y_true[..., 0], [-1])
    y_pred_flat = tf.reshape(y_pred, [-1])
    indices = tf.where(tf.reshape(y_true[..., 1], [-1]) > 0)
    y_pred_flat = tf.cast(y_pred_flat, y_true_flat.dtype)
    equals = tf.gather(y_true_flat, indices) == tf.gather(y_pred_flat, indices)
    equals = tf.cast(equals, tf.float32)
    return tf.reduce_mean(tf.reduce_mean(equals, axis=-1))


class CustomTFMLP(tf.keras.layers.Layer):
    def __init__(self, n_state: int, config: GPT2Config, **kwargs):
        super().__init__(**kwargs)
        self.n_state = n_state
        self.config = config
        self.nx = config.n_embd
        self.c_fc = TFConv1D(
            n_state, self.nx, initializer_range=config.initializer_range, name="c_fc"
        )
        self.c_proj = TFConv1D(
            self.nx, n_state, initializer_range=config.initializer_range, name="c_proj"
        )
        self.act = get_tf_activation(config.activation_function)
        self.dropout = tf.keras.layers.Dropout(config.resid_pdrop)
        self._inputs_history = None
        self._history_enabled = False

    def enable_accumulate_history(self):
        self._inputs_history = []
        self._history_enabled = True

    def disable_accumulate_history(self):
        self._history_enabled = False

    def reset_history(self):
        self._inputs_history = []

    def get_inputs_history(self) -> np.ndarray:
        return np.concatenate(self._inputs_history, axis=0)

    def init(self, base: TFMLP):
        x = tf.random.uniform([1, 1, self.nx], dtype=tf.float16)
        self(x)
        base(x)
        self.set_weights(base.get_weights())

    def call(self, x, training=False, **kwargs):
        self._last_inputs = x

        if self._history_enabled:
            self._inputs_history.append(x.numpy())

        h = self.act(self.c_fc(x))

        h2 = self.c_proj(h)
        h2 = self.dropout(h2, training=training)

        # print(f"x ({x.shape}) => h ({h.shape}) => h2 ({h2.shape})")
        self._last_outputs = h2

        return h2

    def get_activations_stats(self, x: np.ndarray):
        h = self.act(self.c_fc(x))

        h = tf.reshape(h, [-1, self.n_state])
        h_pos_counts = tf.cast(h > 0, tf.int32)
        h_pos_counts = tf.reduce_sum(h_pos_counts, axis=0)
        h_pos_counts = h_pos_counts / tf.shape(h)[0]

        return h_pos_counts.numpy()

    def reduce(
        self,
        counts: np.ndarray,
        x: np.ndarray,
        y: np.ndarray,
        r: int = 1000,
        in_place: bool = False,
    ):
        indices_to_keep = np.argsort(counts)[r:]

        n_state = len(indices_to_keep)
        new_layer = CustomTFMLP(n_state, self.config)
        x_rand = tf.random.uniform([1, 1, self.nx], dtype=tf.float16)
        new_layer(x_rand)

        w, bias = self.c_fc.get_weights()
        w = w[:, indices_to_keep]
        bias = bias[:, indices_to_keep]
        new_layer.c_fc.set_weights([w, bias])

        w, bias = self.c_proj.get_weights()
        w = w[indices_to_keep, :]
        new_layer.c_proj.set_weights([w, bias])

        # for i in range(3):
        #     w, bias = new_layer.c_proj.get_weights()

        #     y1 = y.reshape([-1, self.nx]).astype(np.float32)

        #     y2 = new_layer(x)
        #     y2 = y2.numpy().reshape([-1, self.nx]).astype(np.float32)

        #     y1_mean, y1_std = y1.mean(0), y1.std(0)
        #     y2_mean, y2_std = y2.mean(0), y2.std(0)

        #     bias_correction = y2_mean - y1_mean
        #     scale_correction = y1_std / np.maximum(y2_std, 0.0001)

        #     bias_correction = np.expand_dims(bias_correction, 0)
        #     scale_correction = np.expand_dims(scale_correction, 0)

        #     new_layer.c_proj.set_weights([w * scale_correction, bias - bias_correction])

        if in_place:
            self.n_state = n_state
            self.c_fc = new_layer.c_fc
            self.c_proj = new_layer.c_proj
            return self
        return new_layer

    def reduce_from_history(self, r: int = 10, in_place: bool = False):
        x = self.get_inputs_history()

        print(f"History size: {x.shape}")
        stats = self.get_activations_stats(x)
        y = self(x).numpy()
        new_layer = self.reduce(stats, x=x, y=y, r=r, in_place=in_place)

        y_prim = new_layer(x).numpy()
        mean_error = np.abs((y_prim - y)).mean()
        print(
            f"Reduction error: {mean_error:.5f} relative {mean_error / np.abs(y).mean():.5f} (|y| = {np.abs(y).mean():.5f})"
        )

        return new_layer


class SplitTFBlock(TFBlock):
    def call(
        self,
        x,
        layer_past=None,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        use_cache=False,
        output_attentions=False,
        training=False,
        context_hidden_state=None,
    ):
        a = self.ln_1(x)

        if context_hidden_state is not None:
            context_hidden_state = self.ln_1(context_hidden_state)
            a = tf.concat([context_hidden_state, a], axis=1)

        output_attn = self.attn(
            a,
            layer_past=layer_past,
            attention_mask=attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            use_cache=use_cache,
            output_attentions=output_attentions,
            training=training,
        )
        if context_hidden_state is not None:
            context_size = context_hidden_state.shape[1]
            a = output_attn[0][:, context_size:, :]
            outputs = output_attn[1:]

        else:
            a = output_attn[0]  # output_attn: a, present, (attentions)
            outputs = output_attn[1:]

        x = x + a

        # Cross-Attention Block
        if encoder_hidden_states is not None:
            # add one self-attention block for cross-attention
            if not hasattr(self, "crossattention"):
                raise ValueError(
                    f"If `encoder_hidden_states` are passed, {self} has to be instantiated with "
                    "cross-attention layers by setting `config.add_cross_attention=True`"
                )

            ca = self.ln_cross_attn(x)
            output_cross_attn = self.crossattention(
                ca,
                layer_past=None,
                attention_mask=attention_mask,
                head_mask=head_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                use_cache=False,
                output_attentions=output_attentions,
                training=training,
            )
            ca = output_cross_attn[0]  # output_attn: a, present, (cross_attentions)
            x = x + ca
            outputs = (
                outputs + output_cross_attn[2:]
            )  # add cross attentions if we output attention weights

        m = self.ln_2(x)
        m = self.mlp(m, training=training)
        x = x + m

        outputs = [x] + outputs
        return outputs  # x, present, (attentions, cross_attentions)


@dataclass
class TFBaseModelOutputWithPastAndCrossAttentionsV2(
    TFBaseModelOutputWithPastAndCrossAttentions
):
    input_hidden_states: Optional[Tuple[tf.Tensor]] = None


# @keras_serializable
class SplitTFGPT2MainLayer(tf.keras.layers.Layer):
    config_class = GPT2Config

    def __init__(self, config, *inputs, **kwargs):
        super().__init__(*inputs, **kwargs)

        self.config = config
        self.output_attentions = config.output_attentions
        self.output_hidden_states = config.output_hidden_states
        self.use_cache = config.use_cache
        self.return_dict = config.use_return_dict

        self.num_hidden_layers = config.n_layer
        self.n_embd = config.n_embd
        self.n_positions = config.n_positions
        self.initializer_range = config.initializer_range

        self.wte = TFSharedEmbeddings(
            config.vocab_size,
            config.hidden_size,
            initializer_range=config.initializer_range,
            name="wte",
        )
        self.drop = tf.keras.layers.Dropout(config.embd_pdrop)
        self.h = [
            SplitTFBlock(config, scale=True, name=f"h_._{i}")
            for i in range(config.n_layer)
        ]
        self.ln_f = tf.keras.layers.LayerNormalization(
            epsilon=config.layer_norm_epsilon, name="ln_f"
        )

    def build(self, input_shape):
        with tf.name_scope("wpe"):
            self.wpe = self.add_weight(
                name="embeddings",
                shape=[self.n_positions, self.n_embd],
                initializer=get_initializer(self.initializer_range),
            )

        super().build(input_shape)

    def get_input_embeddings(self):
        return self.wte

    def set_input_embeddings(self, value):
        self.wte.weight = value
        self.wte.vocab_size = shape_list(value)[0]

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer}
        """
        raise NotImplementedError

    @unpack_inputs
    def call(
        self,
        input_ids: Optional[TFModelInputType] = None,
        past_key_values: Optional[Tuple[Tuple[Union[np.ndarray, tf.Tensor]]]] = None,
        attention_mask: Optional[Union[np.ndarray, tf.Tensor]] = None,
        token_type_ids: Optional[Union[np.ndarray, tf.Tensor]] = None,
        position_ids: Optional[Union[np.ndarray, tf.Tensor]] = None,
        head_mask: Optional[Union[np.ndarray, tf.Tensor]] = None,
        inputs_embeds: Optional[Union[np.ndarray, tf.Tensor]] = None,
        encoder_hidden_states: Optional[Union[np.ndarray, tf.Tensor]] = None,
        encoder_attention_mask: Optional[Union[np.ndarray, tf.Tensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        training: Optional[bool] = False,
        context_hidden_states: Optional[Tuple[Union[np.ndarray, tf.Tensor]]] = None,
    ) -> Union[TFBaseModelOutputWithPastAndCrossAttentionsV2, Tuple[tf.Tensor]]:
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time"
            )
        elif input_ids is not None:
            input_shape = shape_list(input_ids)
            input_ids = tf.reshape(input_ids, [-1, input_shape[-1]])
        elif inputs_embeds is not None:
            input_shape = shape_list(inputs_embeds)[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if past_key_values is None:
            past_length = 0
            past_key_values = [None] * len(self.h)
        else:
            past_length = shape_list(past_key_values[0][0])[-2]

        if position_ids is None:
            position_ids = tf.expand_dims(
                tf.range(past_length, input_shape[-1] + past_length), axis=0
            )

        if attention_mask is not None:
            # We create a 3D attention mask from a 2D tensor mask.
            # Sizes are [batch_size, 1, 1, to_seq_length]
            # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
            # this attention mask is more simple than the triangular masking of causal attention
            # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
            attention_mask_shape = shape_list(attention_mask)
            attention_mask = tf.reshape(
                attention_mask, (attention_mask_shape[0], 1, 1, attention_mask_shape[1])
            )

            # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
            # masked positions, this operation will create a tensor which is 0.0 for
            # positions we want to attend and -10000.0 for masked positions.
            # Since we are adding it to the raw scores before the softmax, this is
            # effectively the same as removing these entirely.
            one_cst = tf.constant(1.0)
            attention_mask = tf.cast(attention_mask, dtype=one_cst.dtype)
            attention_mask = tf.multiply(
                tf.subtract(one_cst, attention_mask), tf.constant(-10000.0)
            )

        # Copied from `modeling_tf_t5.py` with -1e9 -> -10000
        if self.config.add_cross_attention and encoder_attention_mask is not None:
            # If a 2D ou 3D attention mask is provided for the cross-attention
            # we need to make broadcastable to [batch_size, num_heads, mask_seq_length, mask_seq_length]
            # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
            encoder_attention_mask = tf.cast(
                encoder_attention_mask, dtype=encoder_hidden_states.dtype
            )
            num_dims_encoder_attention_mask = len(shape_list(encoder_attention_mask))
            if num_dims_encoder_attention_mask == 3:
                encoder_extended_attention_mask = encoder_attention_mask[:, None, :, :]
            if num_dims_encoder_attention_mask == 2:
                encoder_extended_attention_mask = encoder_attention_mask[
                    :, None, None, :
                ]

            # T5 has a mask that can compare sequence ids, we can simulate this here with this transposition
            # Cf. https://github.com/tensorflow/mesh/blob/8d2465e9bc93129b913b5ccc6a59aa97abd96ec6/mesh_tensorflow/transformer/transformer_layers.py#L270
            # encoder_extended_attention_mask = tf.math.equal(encoder_extended_attention_mask,
            #                                         tf.transpose(encoder_extended_attention_mask, perm=(-1, -2)))

            encoder_extended_attention_mask = (
                1.0 - encoder_extended_attention_mask
            ) * -10000.0
        else:
            encoder_extended_attention_mask = None

        encoder_attention_mask = encoder_extended_attention_mask

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        if head_mask is not None:
            raise NotImplementedError
        else:
            head_mask = [None] * self.num_hidden_layers
            # head_mask = tf.constant([0] * self.num_hidden_layers)

        position_ids = tf.reshape(position_ids, [-1, shape_list(position_ids)[-1]])

        if inputs_embeds is None:
            # Note: tf.gather, on which the embedding layer is based, won't check positive out of bound
            # indices on GPU, returning zeros instead. This is a dangerous silent behavior.
            tf.debugging.assert_less(
                input_ids,
                tf.cast(self.config.vocab_size, dtype=input_ids.dtype),
                message=(
                    "input_ids must be smaller than the embedding layer's input dimension (got"
                    f" {tf.math.reduce_max(input_ids)} >= {self.config.vocab_size})"
                ),
            )
            inputs_embeds = self.wte(input_ids, mode="embedding")

        position_embeds = tf.gather(self.wpe, position_ids)

        if token_type_ids is not None:
            token_type_ids = tf.reshape(
                token_type_ids, [-1, shape_list(token_type_ids)[-1]]
            )
            token_type_embeds = self.wte(token_type_ids, mode="embedding")
        else:
            token_type_embeds = tf.constant(0.0)

        position_embeds = tf.cast(position_embeds, dtype=inputs_embeds.dtype)
        token_type_embeds = tf.cast(token_type_embeds, dtype=inputs_embeds.dtype)
        hidden_states = inputs_embeds + position_embeds + token_type_embeds
        hidden_states = self.drop(hidden_states, training=training)

        output_shape = input_shape + [shape_list(hidden_states)[-1]]

        presents = () if use_cache else None
        all_attentions = () if output_attentions else None
        all_cross_attentions = (
            () if output_attentions and self.config.add_cross_attention else None
        )
        all_hidden_states = () if output_hidden_states else None
        input_hidden_states = ()
        for i, (block, layer_past) in enumerate(zip(self.h, past_key_values)):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (
                    tf.reshape(hidden_states, output_shape),
                )

            input_hidden_states += (hidden_states,)

            context_hidden_state = None
            if context_hidden_states is not None:
                context_hidden_state = context_hidden_states[i]

            outputs = block(
                hidden_states,
                layer_past,
                attention_mask,
                head_mask[i],
                encoder_hidden_states,
                encoder_attention_mask,
                use_cache,
                output_attentions,
                training=training,
                context_hidden_state=context_hidden_state,
            )

            hidden_states, present = outputs[:2]
            if use_cache:
                presents = presents + (present,)

            if output_attentions:
                all_attentions = all_attentions + (outputs[2],)
                if (
                    self.config.add_cross_attention
                    and encoder_hidden_states is not None
                ):
                    all_cross_attentions = all_cross_attentions + (outputs[3],)

        hidden_states = self.ln_f(hidden_states)

        hidden_states = tf.reshape(hidden_states, output_shape)
        # Add last hidden state
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if output_attentions:
            # let the number of heads free (-1) so we can extract attention even after head pruning
            attention_output_shape = (
                input_shape[:-1] + [-1] + shape_list(all_attentions[0])[-2:]
            )
            all_attentions = tuple(
                tf.reshape(t, attention_output_shape) for t in all_attentions
            )

        return TFBaseModelOutputWithPastAndCrossAttentionsV2(
            last_hidden_state=hidden_states,
            past_key_values=presents,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
            cross_attentions=all_cross_attentions,
            input_hidden_states=input_hidden_states,
        )


class SplitGPT2Model(TFGPT2PreTrainedModel):
    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        self.transformer = SplitTFGPT2MainLayer(config, name="transformer")

    @classmethod
    def from_pretrained(cls, name, *model_args, **kwargs):
        base_model = TFGPT2Model.from_pretrained(name, *model_args, **kwargs)
        model = super().from_pretrained(name, *model_args, **kwargs)
        model.set_weights(base_model.get_weights())
        return model

    @unpack_inputs
    def call(
        self,
        input_ids: Optional[TFModelInputType] = None,
        past_key_values: Optional[Tuple[Tuple[Union[np.ndarray, tf.Tensor]]]] = None,
        attention_mask: Optional[Union[np.ndarray, tf.Tensor]] = None,
        token_type_ids: Optional[Union[np.ndarray, tf.Tensor]] = None,
        position_ids: Optional[Union[np.ndarray, tf.Tensor]] = None,
        head_mask: Optional[Union[np.ndarray, tf.Tensor]] = None,
        inputs_embeds: Optional[Union[np.ndarray, tf.Tensor]] = None,
        encoder_hidden_states: Optional[Union[np.ndarray, tf.Tensor]] = None,
        encoder_attention_mask: Optional[Union[np.ndarray, tf.Tensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        training: Optional[bool] = False,
        context_size: Optional[int] = None,
    ) -> Union[TFBaseModelOutputWithPastAndCrossAttentionsV2, Tuple[tf.Tensor]]:
        r"""
        encoder_hidden_states  (`tf.Tensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention if
            the model is configured as a decoder.
        encoder_attention_mask (`tf.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on the padding token indices of the encoder input. This mask is used in
            the cross-attention if the model is configured as a decoder. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

        past_key_values (`Tuple[Tuple[tf.Tensor]]` of length `config.n_layers`)
            contains precomputed key and value hidden states of the attention blocks. Can be used to speed up decoding.
            If `past` are used, the user can optionally input only the last `decoder_input_ids` (those that don't have
            their past key value states given to this model) of shape `(batch_size, 1)` instead of all
            `decoder_input_ids` of shape `(batch_size, sequence_length)`.
        use_cache (`bool`, *optional*, defaults to `True`):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past`). Set to `False` during training, `True` during generation
        """

        if context_size is None:
            outputs = self.transformer(
                input_ids=input_ids,
                past_key_values=past_key_values,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=True,
                return_dict=return_dict,
                training=training,
            )

            return outputs
        else:
            assert context_size * 2 == input_ids.shape[1]
            context_input_ids = input_ids[:, :context_size]
            content_input_ids = input_ids[:, context_size:]
            context_outputs = self.transformer(
                input_ids=context_input_ids,
                past_key_values=past_key_values,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=True,
                return_dict=False,
                training=training,
            )
            context_hidden_states = context_outputs.input_hidden_states

            position_ids = tf.expand_dims(
                tf.range(context_size, 2 * context_size), axis=0
            )
            # context_hidden_states = [tf.stop_gradient(c) for c in context_hidden_states]

            outputs = self.transformer(
                input_ids=content_input_ids,
                past_key_values=past_key_values,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=False,
                return_dict=False,
                training=training,
                context_hidden_states=context_hidden_states,
            )

            return context_outputs, outputs
