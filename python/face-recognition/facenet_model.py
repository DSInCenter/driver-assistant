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


def get_facenet_model():
    layers = [
        keras.layers.InputLayer(input_shape=[112, 112, 1]),
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


model = get_facenet_model()
model.summary()
