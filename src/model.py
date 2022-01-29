import tensorflow as tf

class Generator(tf.keras.layers.Layer):
    """Generator network.
    """
    def __init__(self, input_dim, output_dim, nb_layers=2, nb_units=256, concat_every_fcl=True, batchnorm=False):  
        super(Generator, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.nb_layers = nb_layers
        self.nb_units = nb_units
        self.concat_every_fcl = concat_every_fcl
        self.batchnorm = batchnorm
        self.layers = []

    def build(self, input_shape):
        """Builds the FC stacks."""
        for i in range(self.nb_layers):
            fc_layer = tf.keras.layers.Dense(
                self.nb_units,
                activation = None,
                kernel_regularizer = tf.keras.regularizers.L2(2.5e-5)
            )   
            norm_layer = tf.keras.layers.BatchNormalization()
            self.layers.append([fc_layer, norm_layer])

        super(Generator, self).build(input_shape)

    def call(self, inputs):
        """Return the output of the Generator.
        Args:
            inputs: tensor with shape [batch_size, input_dim + nb_classes]
        Returns:
            Output of Generator.
            float32 tensor with shape [batch_size, output_dim]
        """
        y = inputs[:,self.input_dim:]
        for i, all_layers in enumerate(self.layers[:-1]):
            # Run inputs through the sublayers.
            fc_layer, norm_layer = all_layers
            with tf.name_scope("g_layer_%d" % i):
                x = fc_layer(inputs) if i==0 else fc_layer(x)
                if self.batchnorm:
                    x = norm_layer(x)
                x = tf.keras.activations.LeakyReLU(alpha=0.2)(x)
                if self.concat_every_fcl:
                    x = tf.keras.layers.concatenate([x,y],axis=1)
        fc_layer, norm_layer = self.layers[-1]
        with tf.name_scope("g_layer_ouput"):
            output = tf.keras.layers.Dense(self.output_dim)(x)
            # No activation func at last layer
            #x = tf.keras.activations.LeakyReLU(alpha=0.2)(x)
        return output

class Encoder(tf.keras.layers.Layer):
    """Encoder (H) network.
    """
    def __init__(self, input_dim, output_dim, feat_dim, nb_layers=2, nb_units=256, batchnorm=False):  
        super(Encoder, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.feat_dim = feat_dim
        self.nb_layers = nb_layers
        self.nb_units = nb_units
        self.batchnorm = batchnorm
        self.layers = []

    def build(self, input_shape):
        """Builds the FC stacks."""
        for i in range(self.nb_layers):
            fc_layer = tf.keras.layers.Dense(
                self.nb_units,
                activation = None
            )   
            norm_layer = tf.keras.layers.BatchNormalization()

            self.layers.append([fc_layer, norm_layer])

        super(Encoder, self).build(input_shape)

    def call(self, inputs):
        """Return the output of the Encoder(H network).
        Args:
            inputs: tensor with shape [batch_size, input_dim]
        Returns:
            Output of Encoder.
            float32 tensor with shape [batch_size, output_dim]
        """
        for i, all_layers in enumerate(self.layers[:-1]):
            # Run inputs through the sublayers.
            fc_layer, norm_layer = all_layers
            with tf.name_scope("h_layer_%d" % i):
                x = fc_layer(inputs) if i==0 else fc_layer(x)
                if self.batchnorm:
                    x = norm_layer(x)
                x = tf.keras.activations.LeakyReLU(alpha=0.2)(x)
        fc_layer, norm_layer = self.layers[-1]
        with tf.name_scope("h_layer_ouput"):
            output = tf.keras.layers.Dense(self.output_dim)(x)
            logits = output[:, self.feat_dim:]
            y = tf.nn.softmax(logits)
        return output, y
    
class Discriminator(tf.keras.layers.Layer):
    """Discriminator network.
    """
    def __init__(self, input_dim, output_dim, name, nb_layers=2, nb_units=256, batchnorm=True):  
        super(Discriminator, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.name = name
        self.nb_layers = nb_layers
        self.nb_units = nb_units
        self.batchnorm = batchnorm
        self.layers = []

    def build(self, input_shape):
        """Builds the FC stacks."""
        for i in range(self.nb_layers):
            fc_layer = tf.keras.layers.Dense(
                self.nb_units,
                activation = None
            )   
            norm_layer = tf.keras.layers.BatchNormalization()

            self.layers.append([fc_layer, norm_layer])

        super(Discriminator, self).build(input_shape)

    def call(self, inputs):
        """Return the output of the Discriminator network.
        Args:
            inputs: tensor with shape [batch_size, input_dim]
        Returns:
            Output of Discriminator.
            float32 tensor with shape [batch_size, 1]
        """
        fc_layer, norm_layer = self.layers[0]
        with tf.name_scope("%s_layer_0" % self.name):
            x = fc_layer(inputs)
            x = tf.keras.activations.LeakyReLU(alpha=0.2)(x)
            
        for i, all_layers in enumerate(self.layers[1:-1]):
            # Run inputs through the sublayers.
            fc_layer, norm_layer = all_layers
            with tf.name_scope("%s_layer_%d" % (self.name,i+1)):
                x = fc_layer(x)
                if self.batchnorm:
                    x = norm_layer(x)
                x = tf.keras.activations.tanh(x)
                #x = tf.keras.activations.LeakyReLU(alpha=0.2)(x)
        fc_layer, norm_layer = self.layers[-1]
        with tf.name_scope("%s_layer_ouput" % self.name):
            output = tf.keras.layers.Dense(1)(x)
        return output


class ConvModule(tf.keras.layers.Layer):
    """Convolution Module.
    """
    def __init__(self, params):  
        super(ConvModule, self).__init__()
        self.params = params
        self.layers = []

    def build(self, input_shape):
        """Builds the convolution stack."""
        params = self.params
        for i in range(params["num_cb"]):
            if i == 0:
                conv_layer = tf.keras.layers.Conv2D(
                    filters = params["num_channels"]//2,
                    kernel_size = (15,1), strides = (1, 1),
                    padding = 'same',activation = None)
                rconv_layer = tf.keras.layers.Conv2D(
                    filters = params["num_channels"]//2,
                    kernel_size = (1,1), strides = (1, 1),
                    padding = 'same')
            else:
                conv_layer = tf.keras.layers.Conv2D(
                    filters = params["num_channels"],
                    kernel_size = (5,1), strides = (1, 1),
                    padding = 'same',activation = None)   
                rconv_layer = tf.keras.layers.Conv2D(
                    filters = params["num_channels"],
                    kernel_size = (1,1), strides = (1, 1),
                    padding = 'same')
            pooling_layer = tf.keras.layers.MaxPool2D(pool_size = (2, 1))
            norm_layer = tf.keras.layers.BatchNormalization()

            self.layers.append([conv_layer, rconv_layer, pooling_layer, norm_layer])

        super(ConvModule, self).build(input_shape)

    def get_config(self):
        return {
                "params": self.params,
        }

    def call(self, inputs):
        """Return the output of the convolution module.
        Args:
            inputs: tensor with shape [batch_size, input_length*2^num_cb, 1, 4]
            training: boolean, whether in training mode or not.
        Returns:
            Output of convolution module.
            float32 tensor with shape [batch_size, input_length, num_channels]
        """

        for i, layer in enumerate(self.layers):
            # Run inputs through the sublayers.
            conv_layer, rconv_layer, pooling_layer, norm_layer = layer
            with tf.name_scope("layer_%d" % i):
                #x = tf.keras.activations.gelu(inputs) if i == 0 else tf.keras.activations.gelu(x)
                x = tf.keras.activations.relu(inputs) if i == 0 else tf.keras.activations.relu(x)
                x = conv_layer(x)
                x = pooling_layer(x)
                tmp = rconv_layer(x)
                x = x + tmp
                x = norm_layer(x)
        return tf.squeeze(x, axis = 2)