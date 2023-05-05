import tensorflow as tf


def masked_lm_loss(y_true: tf.Tensor, logits: tf.Tensor) -> tf.Tensor:
    y_true_labels = y_true[..., 0]
    y_true_mask = y_true[..., 1]
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction="none"
    )
    masked_loss = loss_fn(y_true_labels, logits) * tf.cast(y_true_mask, logits.dtype)
    return tf.reduce_mean(tf.reduce_mean(masked_loss, axis=-1))


def masked_accuracy(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    y_true_flat = tf.reshape(y_true[..., 0], [-1])
    y_pred_flat = tf.reshape(y_pred, [-1])
    indices = tf.where(tf.reshape(y_true[..., 1], [-1]) > 0)
    y_pred_flat = tf.cast(y_pred_flat, y_true_flat.dtype)
    equals = tf.gather(y_true_flat, indices) == tf.gather(y_pred_flat, indices)
    equals = tf.cast(equals, tf.float32)
    return tf.reduce_mean(tf.reduce_mean(equals, axis=-1))
