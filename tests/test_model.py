import tensorflow as tf

import tf_nano_gpt.model as m


class TestModel(tf.test.TestCase):
    def setUp(self):
        ...

    def test_gelu(self):
        x = tf.random.uniform((100, 10)) - 0.5
        y = tf.nn.gelu(x)

        self.assertAllEqual(y >= -0.155, tf.cast(tf.ones_like(y), tf.bool))

    def test_MLP(self):

        mlp = m.MLP(10, bias=True)

        x = tf.random.uniform((7, 16, 8))
        y = mlp(x)
        self.assertAllEqual(y.shape, (7, 16, 10))

        # checking initialization
        kernel0, bias0 = mlp.c_proj.get_weights()
        mlp.init_weights(10)
        kernel1, bias1 = mlp.c_proj.get_weights()
        self.assertNotAllClose(kernel0, kernel1)
        self.assertAllClose(bias0, bias1)

    def test_CasualSelfAttentionBlock(self):

        block = m.CasualSelfAttentionBlock(128, 8, dropout=0.5)

        x = tf.random.uniform((7, 16, 128))
        y = block(x, x)

        self.assertAllEqual(y.shape, (7, 16, 128))

    def test_TokenAndPositionEmbedding(self):

        layer = m.TokenAndPositionEmbedding(32, 256, 128, dropout=0.1)

        x = tf.random.uniform((7, 32), maxval=256, dtype=tf.int32)
        y = layer(x)

        self.assertAllEqual(y.shape, (7, 32, 128))

    def test_GPT(self):

        config = m.GPTConfig(128, 512, n_embd=128)
        model = m.GPT(config)

        x = tf.random.uniform((7, 32), maxval=256, dtype=tf.int32)
        y = model(x)

        self.assertAllEqual(y.shape, (7, 32, 512))
