import keras
import tensorflow as tf
from transformers.modeling_tf_utils import TFConv1D

import tf_nano_gpt.layers as nl


class TestLayers(tf.test.TestCase):
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
