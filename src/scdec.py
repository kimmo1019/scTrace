import tensorflow as tf
from model import Generator, Discriminator, Encoder
import numpy as np
from util import scATAC_Sampler, scRNA_Sampler, Joint_Sampler
from util import Mixture_sampler
import sys

class scDEC(object):
    """scDEC model for clustering.
    """
    def __init__(self, params):  
        super(scDEC, self).__init__()
        self.params = params
        self.g_net = Generator(input_dim=params['z_dim'], output_dim = params['x_dim'],nb_layers=10, nb_units=512, concat_every_fcl=False)
        self.h_net = Encoder(input_dim=params['x_dim'], output_dim = params['z_dim']+params['nb_classes'],feat_dim=params['z_dim'],nb_layers=10,nb_units=256)
        self.dz_net = Discriminator(input_dim=params['z_dim'],name='dz_net',nb_layers=2,nb_units=256)
        self.dx_net = Discriminator(input_dim=params['x_dim'],name='dx_net',nb_layers=2,nb_units=256)
        self.g_h_optimizer = tf.keras.optimizers.Adam(params['lr'], beta_1=0.5, beta_2=0.9)
        self.d_optimizer = tf.keras.optimizers.Adam(params['lr'], beta_1=0.5, beta_2=0.9)
        self.z_sampler = Mixture_sampler(nb_classes=params['nb_classes'],N=10000,dim=params['x_dim'],sd=1)
        if params['data_type'] == 'scRNA-seq':
            self.x_sampler = scRNA_Sampler(params)
        elif params['data_type'] == 'scATAC-seq':
            self.x_sampler = scATAC_Sampler(params)
        elif params['data_type'] == 'Joint':
            self.x_sampler = Joint_Sampler(params)
        else:
            print("Data type error")
            sys.exit()
            
    def get_config(self):
        return {
                "params": self.params,
        }
    
    def discriminator_loss(self, real, generated):
        loss_ce = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        real_loss = loss_ce(tf.ones_like(real), real)
        generated_loss = loss_ce(tf.zeros_like(generated), generated)
        total_disc_loss = real_loss + generated_loss
        return total_disc_loss * 0.5
    
    def generator_loss(self, generated):
        loss_ce = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        return loss_ce(tf.ones_like(generated), generated)
    
    def mse_loss(self, data1, data2):
        loss = tf.reduce_mean(tf.abs(data1 - data2))
        return loss

    def train(self, inputs):
        """train scDEC model.
        Args:
            inputs: input tensor list of 2
                First item:  latent tensor with shape [batch_size, z_dim].
                Second item: latent onehot tensor with shape [batch_size, nb_classes].
                Third item: obervation data with shape [batch_size, x_dim].
            training: boolean, whether in training mode or not.
        Returns:
                returns a dictionary {
                    outputs: int tensor with shape [batch_size, decoded_length]
                    scores: float tensor with shape [batch_size]}
        """
        @tf.function
        def train_step(data_z, data_z_onehot, data_x):
            with tf.GradientTape(persistent=True) as tape:
                data_z, data_z_onehot, data_x = inputs[0], inputs[1], inputs[2]
                data_z_combine = tf.keras.layers.Concatenate([data_z, data_z_onehot], axis=-1)
                data_x_fake = self.g_net(data_z_combine)
                data_z_latent_fake, data_z_onehot_fake = self.h_net(data_x)
                data_z_fake = data_z_latent_fake[:,:self.params['x_dim']]
                
                
                
            
            
        
        
