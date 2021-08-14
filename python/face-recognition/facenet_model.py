import tensorflow as tf
import tensorflow.keras as keras


class ResidualBlock(keras.layers.Layer):
    def __init__(self, output_channels, expansion, index, stride=1, activation='prelu', **kwargs):
        super().__init__(name=name_generator_residual(expansion, stride, output_channels, index), **kwargs)
        self.stride = stride
        self.output_channels = output_channels
        self.activation = activation
        # self.activation = keras.activations.relu
        self.main_layers = [
            keras.layers.Conv2D(expansion, kernel_size=1, strides=1, use_bias=False,
                                name='expansion{}_{}'.format(index, expansion)),
            keras.layers.BatchNormalization(),
            keras.layers.ReLU(),
            keras.layers.DepthwiseConv2D(kernel_size=3, strides=stride,
                                         padding='same', use_bias=False,
                                         name='depthwise{}_{}'.format(index, stride)),
            keras.layers.BatchNormalization(),
            keras.layers.ReLU(),
            keras.layers.Conv2D(kernel_size=1, filters=output_channels, use_bias=False,
                                name='conv{}'.format(index)),
            keras.layers.BatchNormalization()
        ]

    def call(self, inputs, **kwargs):
        Z = inputs
        for layer in self.main_layers:
            # print(layer.name, flush=True)
            Z = layer(Z)

        return inputs + Z if self.stride == 1 else Z


def name_generator_residual(expansion, stride, output, index):
    return "residual_block{}_e{}_s{}_c{}".format(index, expansion, stride, output)


class TripletDistanceLayer(keras.layers.Layer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, anchor, positive, negative):
        pos_distance = tf.reduce_sum(tf.square(anchor - positive), -1)
        neg_distance = tf.reduce_sum(tf.square(anchor - negative), -1)
        return pos_distance, neg_distance


class SiameseModel(keras.models.Model):
    """The Siamese Network model with a custom training and testing loops.

    Computes the triplet loss using the three embeddings produced by the
    Siamese Network.

    The triplet loss is defined as:
        L(A, P, N) = max(‖f(A) - f(P)‖² - ‖f(A) - f(N)‖² + margin, 0)
    """

    def __init__(self, siamese_network, margin=0.5):
        super(SiameseModel, self).__init__()
        self.siamese_network = siamese_network
        self.margin = margin
        self.loss_tracker = keras.metrics.Mean(name='loss')

    def call(self, inputs):
        return self.siamese_network(inputs)

    def train_step(self, data):
        with tf.GradientTape() as tape:
            loss = self._compute_loss(data)

        gradients = tape.gradient(loss, self.siamese_network.trainable_weights)
        self.optimizer.apply_gradients(
            zip(gradients, self.siamese_network.trainable_weights)
        )

        self.loss_tracker.update_state(loss)
        return {'loss', self.loss_tracker.result()}

    def test_step(self, data):
        loss = self._compute_loss(data)
        self.loss_tracker.update_state(loss)
        return {'loss', self.loss_tracker.result()}

    def _compute_loss(self, data):
        pos_distance, neg_distance = self.siamese_network(data)
        return tf.maximum(pos_distance - neg_distance + self.margin, 0.0)

    @property
    def metrics(self):
        return [self.loss_tracker]


def get_facenet_model():
    layers = [
        keras.layers.InputLayer(input_shape=[112, 112, 3]),
        keras.layers.Conv2D(filters=64, strides=2, kernel_size=3, padding='same', use_bias=False),
        keras.layers.BatchNormalization(),
        keras.layers.ReLU(),
        keras.layers.DepthwiseConv2D(strides=1, kernel_size=3, padding='same', use_bias=False),
        keras.layers.BatchNormalization(),
        keras.layers.ReLU()
    ]
    i = 1
    layers += [ResidualBlock(output_channels=64, stride=2, expansion=128, index=i)]
    for i in range(2, 6):
        layers += [ResidualBlock(output_channels=64, stride=1, expansion=128, index=i)]

    layers += [ResidualBlock(output_channels=128, index=6, stride=2, expansion=256)]
    for i in range(7, 13):
        layers += [ResidualBlock(output_channels=128, index=i, stride=1, expansion=256)]

    layers += [ResidualBlock(output_channels=128, expansion=512, stride=2, index=13)]

    for i in range(14, 16):
        layers += [ResidualBlock(output_channels=128, stride=1, expansion=256, index=i)]

    layers += [
        keras.layers.Conv2D(filters=512, kernel_size=1, strides=1, use_bias=False),
        keras.layers.BatchNormalization(),
        keras.layers.DepthwiseConv2D(kernel_size=7, strides=1, use_bias=False),
        keras.layers.BatchNormalization(),
        keras.layers.Conv2D(filters=128, kernel_size=1, strides=1)
    ]

    return keras.models.Sequential(layers)

# model = get_facenet_model()
# model.summary()
