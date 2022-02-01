import tensorflow as tf

class Generator(tf.keras.Model):
    """Generator network.
    """
    def __init__(self, input_dim, z_dim, output_dim, nb_layers=2, nb_units=256, concat_every_fcl=True, batchnorm=False):  
        super(Generator, self).__init__()
        self.input_layer = tf.keras.layers.Input((input_dim,))
        self.input_dim = input_dim
        self.z_dim = z_dim
        self.output_dim = output_dim
        self.nb_layers = nb_layers
        self.nb_units = nb_units
        self.concat_every_fcl = concat_every_fcl
        self.batchnorm = batchnorm
        self.all_layers = []
        
        """Builds the FC stacks."""
        for i in range(self.nb_layers):
            units = self.output_dim if i == self.nb_layers-1 else self.nb_units
            fc_layer = tf.keras.layers.Dense(
                units = units,
                activation = None,
                kernel_regularizer = tf.keras.regularizers.L2(2.5e-5)
            )   
            norm_layer = tf.keras.layers.BatchNormalization()
            self.all_layers.append([fc_layer, norm_layer])
        self.out = self.call(self.input_layer)

    def call(self, inputs, training=True):
        """Return the output of the Generator.
        Args:
            inputs: tensor with shape [batch_size, z_dim + nb_classes]
        Returns:
            Output of Generator.
            float32 tensor with shape [batch_size, output_dim]
        """
        y = inputs[:,self.z_dim:]
        for i, layers in enumerate(self.all_layers[:-1]):
            # Run inputs through the sublayers.
            fc_layer, norm_layer = layers
            with tf.name_scope("g_layer_%d" % i):
                x = fc_layer(inputs) if i==0 else fc_layer(x)
                if self.batchnorm:
                    x = norm_layer(x)
                x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
                if self.concat_every_fcl:
                    x = tf.keras.layers.concatenate([x,y],axis=1)
        fc_layer, norm_layer = self.all_layers[-1]
        with tf.name_scope("g_layer_ouput"):
            output = fc_layer(x)
            #utput = tf.keras.layers.Dense(self.output_dim)(x)
            # No activation func at last layer
            #x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
        return output

class Encoder(tf.keras.Model):
    """Encoder (H) network.
    """
    def __init__(self, input_dim, output_dim, feat_dim, nb_layers=2, nb_units=256, batchnorm=False):  
        super(Encoder, self).__init__()
        self.input_layer = tf.keras.layers.Input((input_dim,))
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.feat_dim = feat_dim
        self.nb_layers = nb_layers
        self.nb_units = nb_units
        self.batchnorm = batchnorm
        self.all_layers = []
        """Builds the FC stacks."""
        for i in range(self.nb_layers):
            units = self.output_dim if i == self.nb_layers-1 else self.nb_units
            fc_layer = tf.keras.layers.Dense(
                units = units,
                activation = None
            )   
            norm_layer = tf.keras.layers.BatchNormalization()
            self.all_layers.append([fc_layer, norm_layer])
        self.out = self.call(self.input_layer)

    def call(self, inputs, training=True):
        """Return the output of the Encoder(H network).
        Args:
            inputs: tensor with shape [batch_size, input_dim]
        Returns:
            Output of Encoder.
            float32 tensor with shape [batch_size, output_dim]
        """
        for i, layers in enumerate(self.all_layers[:-1]):
            # Run inputs through the sublayers.
            fc_layer, norm_layer = layers
            with tf.name_scope("h_layer_%d" % i):
                x = fc_layer(inputs) if i==0 else fc_layer(x)
                if self.batchnorm:
                    x = norm_layer(x)
                x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
        fc_layer, norm_layer = self.all_layers[-1]
        with tf.name_scope("h_layer_ouput"):
            output = fc_layer(x)
            logits = output[:, self.feat_dim:]
            y = tf.nn.softmax(logits)
        return output, y
    
class Discriminator(tf.keras.Model):
    """Discriminator network.
    """
    def __init__(self, input_dim, model_name, nb_layers=2, nb_units=256, batchnorm=True):  
        super(Discriminator, self).__init__()
        self.input_layer = tf.keras.layers.Input((input_dim,))
        self.input_dim = input_dim
        self.model_name = model_name
        self.nb_layers = nb_layers
        self.nb_units = nb_units
        self.batchnorm = batchnorm
        self.all_layers = []
        """Builds the FC stacks."""
        for i in range(self.nb_layers):
            units = 1 if i == self.nb_layers-1 else self.nb_units
            fc_layer = tf.keras.layers.Dense(
                units = units,
                activation = None
            )
            norm_layer = tf.keras.layers.BatchNormalization()

            self.all_layers.append([fc_layer, norm_layer])
        self.out = self.call(self.input_layer)

    def call(self, inputs, training=True):
        """Return the output of the Discriminator network.
        Args:
            inputs: tensor with shape [batch_size, input_dim]
        Returns:
            Output of Discriminator.
            float32 tensor with shape [batch_size, 1]
        """
        fc_layer, norm_layer = self.all_layers[0]
        with tf.name_scope("%s_layer_0" % self.model_name):
            x = fc_layer(inputs)
            x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
            
        for i, layers in enumerate(self.all_layers[1:-1]):
            # Run inputs through the sublayers.
            fc_layer, norm_layer = layers
            with tf.name_scope("%s_layer_%d" % (self.model_name,i+1)):
                x = fc_layer(x)
                if self.batchnorm:
                    x = norm_layer(x)
                x = tf.keras.activations.tanh(x)
                #x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
        fc_layer, norm_layer = self.all_layers[-1]
        with tf.name_scope("%s_layer_ouput" % self.model_name):
            output = fc_layer(x)
        return output