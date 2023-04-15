import keras
import tensorflow as tf
from transformers import GPT2Config
from transformers.modeling_tf_utils import TFConv1D
from transformers.models.gpt2.modeling_tf_gpt2 import TFAttention
from wise_python.logging import measuretime

import tf_nano_gpt.layers as nl


class TestLayers(tf.test.TestCase):
    def setUp(self):
        ...

    def test_FactorizedConv1D_exact(self):
        x = tf.random.uniform((100, 5, 786)) - 0.5

        conv1d = nl.FactorizedConv1D(256, top_k=256)

        dense_conv1d = tf.keras.layers.Conv1D(256, 1)
        y1 = dense_conv1d(x)

        conv1d.init_from_conv1d(dense_conv1d)
        y2 = conv1d(x)

        self.assertAllClose(y1, y2, atol=1e-5)

    def test_FactorizedConv1D_approx(self):
        x = tf.random.uniform((100, 5, 786)) - 0.5

        conv1d = nl.FactorizedConv1D(256, top_k=32)

        dense_conv1d = tf.keras.layers.Conv1D(256, 1)
        y1 = dense_conv1d(x)

        conv1d.init_from_conv1d(dense_conv1d)
        y2 = conv1d(x)
        self.assertAllEqual(y1.shape, y2.shape)

    def test_LoRADense(self):
        x = tf.random.uniform((100, 786)) - 0.5
        base = tf.keras.layers.Dense(256)
        lora = nl.LoRADense(base, 8)

        y1 = base(x)
        y2 = lora(x)

        self.assertAllClose(y1, y2, atol=1e-5)

        model1 = keras.Sequential([base])
        model1(x)
        model1.summary()

        model2 = keras.Sequential([lora])
        model2(x)
        model2.summary()

    def test_LoRADense_with_TFConv1D(self):
        x = tf.random.uniform((100, 5, 786)) - 0.5
        base = TFConv1D(256, 786)
        lora = nl.LoRADense(base, 8)

        y1 = base(x)
        y2 = lora(x)

        self.assertAllClose(y1, y2, atol=1e-5)

        model1 = keras.Sequential([base])
        model1(x)
        model1.summary()

        model2 = keras.Sequential([lora])
        model2(x)
        model2.summary()

    def test_ApproxTFAttention(self):
        config = GPT2Config.from_pretrained("gpt2")
        nx = config.n_embd
        layer = TFAttention(nx, config, scale=True)
        approx_layer = nl.ApproxTFAttention(nx, config, scale=True)

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

        x = tf.random.uniform((4, 512, 768)) - 0.5
        # build weights
        layer(x, **args)
        approx_layer(x, **args)
        approx_layer.set_weights(layer.get_weights())

        layer(x, **args)[0]
        approx_layer(x, **args)[0]
        # # self.assertAllClose(y1, y2)

        with measuretime("baseline"):
            for _ in range(20):
                layer(x, **args)

        with measuretime("casual"):
            for _ in range(20):
                approx_layer(x, **args)

    def test_casual_softmax(self):
        x = tf.random.uniform((4, 12, 512, 512)) - 0.5
        x = nl.ApproxTFAttention.apply_casual_attention_mask(x)

        casual_softmax = tf.function(nl.casual_softmax)
        y1 = tf.nn.softmax(x, axis=-1)
        y2 = casual_softmax(x)

        self.assertAllClose(y2, tf.linalg.LinearOperatorLowerTriangular(y2).to_dense())

        self.assertAllClose(y1, y2)

        with measuretime("baseline"):
            for _ in range(100):
                tf.nn.softmax(x, axis=-1)

        with measuretime("casual"):
            for _ in range(100):
                casual_softmax(x)

    def test_TopKFrozenEmbeddings(self):
        embeddings = tf.random.normal((20000, 512))
        input_ids = tf.random.uniform((57, 3), minval=0, maxval=20000, dtype=tf.int32)

        layer = nl.TopKFrozenEmbeddings()
        layer.set_embeddings_matrix(embeddings)

        y = layer(input_ids)
        self.assertEqual(y.shape, (57, 3, 512))

        x = tf.random.uniform((57, 512)) - 0.5
        probs1, indices1 = layer(x, mode="linear")
        self.assertEqual(probs1.shape, (57,))
        self.assertEqual(indices1.shape, (57,))

        probs2, indices2 = layer(y, mode="linear")
        self.assertEqual(probs2.shape, (57, 3))
        self.assertEqual(indices2.shape, (57, 3))

        self.assertAllClose(indices2, input_ids)

        logits = layer(y, mode="test")
        self.assertEqual(logits.shape, (57, 3, 20000))

        y_true = tf.stack([input_ids, tf.ones_like(input_ids)], -1)

        loss = layer.get_loss(probs2, indices2, y_true)
        self.assertAllClose(loss, 0.0)

        y_true = tf.stack([input_ids + 1, tf.ones_like(input_ids)], -1)
        loss = layer.get_loss(probs2, indices2, y_true)
        # loss / 15 should be of order 1
        self.assertAllClose(
            loss / 15, tf.losses.binary_crossentropy([1.0], [0.0]) / 15, atol=0.01
        )
