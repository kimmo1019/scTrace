import tensorflow as tf
from model import Generator, Discriminator, Encoder
import numpy as np
from sklearn.metrics.cluster import normalized_mutual_info_score, adjusted_rand_score,homogeneity_score
from util import scATAC_Sampler, scRNA_Sampler, Joint_Sampler
from util import Mixture_sampler
import dateutil.tz
import datetime
import sys
import copy
import os

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
        
        now = datetime.datetime.now(dateutil.tz.tzlocal())
        self.timestamp = now.strftime('%Y%m%d_%H%M%S')
        
        self.checkpoint_path = "checkpoints/train"
        if not os.path.exists(self.checkpoint_path):
            os.makedirs(self.checkpoint_path)
        
        self.save_dir = "results/%s" % self.timestamp
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)   
        ckpt = tf.train.Checkpoint(g_net = self.g_net,
                                   h_net = self.h_net,
                                   dz_net = self.dz_net,
                                   dx_net = self.dx_net,
                                   g_h_optimizer = self.g_h_optimizer,
                                   d_optimizer = self.d_optimizer)
        self.ckpt_manager = tf.train.CheckpointManager(ckpt, self.checkpoint_path, max_to_keep=100)                 

        if self.ckpt_manager.latest_checkpoint:
            ckpt.restore(self.ckpt_manager.latest_checkpoint)
        print ('Latest checkpoint restored!!')
        
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
    
    def mse_loss(self, data1, data2):
        loss = tf.reduce_mean(tf.abs(data1 - data2))
        return loss


    @tf.function
    def train_step(self, data_z, data_z_onehot, data_x, update=0):
        """Update scDEC model given a batch of data.
        Args:
            inputs: input tensor list of 4
                First item:  latent tensor with shape [batch_size, z_dim].
                Second item: latent onehot tensor with shape [batch_size, nb_classes].
                Third item: obervation data with shape [batch_size, x_dim].
                Fourth item: 0: update generators, 1: update discriminators
        Returns:
                returns various of loss functions.
        """
        with tf.GradientTape(persistent=True) as tape:
            data_z_combine = tf.keras.layers.Concatenate([data_z, data_z_onehot], axis=-1)
            data_x_ = self.g_net(data_z_combine)
            data_z_latent_, data_z_onehot_ = self.h_net(data_x)
            data_z_ = data_z_latent_[:,:self.params['z_dim']]
            data_z_logits_ = data_z_latent_[:,self.params['z_dim']:]
            
            data_z_latent__, data_z_onehot__ = self.h_net(data_x_)
            data_z__ = data_z_latent__[:,:self.params['z_dim']]
            data_z_logits__ = data_z_latent__[:,self.params['z_dim']:]
            
            data_z_combine_ = tf.keras.layers.Concatenate([data_z_, data_z_onehot_])
            data_x__ = self.g_net(data_z_combine_)
            
            data_dx_ = self.dx_net(data_x_)
            data_dz_ = self.dz_net(data_z_)
            
            l2_loss_x = tf.reduce_mean((data_x - data_x__)**2)
            l2_loss_z = tf.reduce_mean((data_z - data_z__)**2)
            
            CE_loss_z = tf.nn.softmax_cross_entropy_with_logits(labels=data_z_onehot, logits=data_z_logits__)
            
            g_loss_adv = -tf.reduce_mean(data_dx_)
            h_loss_adv = -tf.reduce_mean(data_dz_)
            g_h_loss = g_loss_adv+h_loss_adv+self.params['alpha']*(l2_loss_x + l2_loss_z)+ \
                        self.params['beta']*CE_loss_z
            
            data_dx = self.dx_net(data_x)
            data_dz = self.dz_net(data_z)
            
            dz_loss = -tf.reduce_mean(data_dz) + tf.reduce_mean(data_dz_)
            dx_loss = -tf.reduce_mean(data_dx) + tf.reduce_mean(data_dx_)
            
            #gradient penalty for z
            epsilon_z = tf.random.uniform([],minval=0., maxval=1.)
            data_z_hat = data_z*epsilon_z + data_z_*(1-epsilon_z)
            data_dz_hat = self.dz_net(data_z_hat)
            grad_z = tf.gradients(data_dz_hat, data_z_hat)[0] #(bs,z_dim)
            grad_norm_z = tf.sqrt(tf.reduce_sum(tf.square(grad_z), axis=1))#(bs,) 
            gpz_loss = tf.reduce_mean(tf.square(grad_norm_z - 1.0))
            
            #gradient penalty for x
            epsilon_x = tf.random.uniform([],minval=0., maxval=1.)
            data_x_hat = data_x*epsilon_x + data_x_*(1-epsilon_x)
            data_dx_hat = self.dx_net(data_x_hat)
            grad_x = tf.gradients(data_dx_hat, data_x_hat)[0] #(bs,x_dim)
            grad_norm_x = tf.sqrt(tf.reduce_sum(tf.square(grad_x), axis=1))#(bs,) 
            gpx_loss = tf.reduce_mean(tf.square(grad_norm_x - 1.0))
            
            d_loss = dx_loss+dz_loss+self.params['gama']*(gpz_loss+gpx_loss)
        
        # Calculate the gradients for generators and discriminators
        g_h_gradients = tape.gradient(g_h_loss, self.g_net.trainable_variables+self.h_net.trainable_variables)
        d_gradients = tape.gradient(d_loss, self.dz_net.trainable_variables+self.dx_net.trainable_variables)
        
        # Apply the gradients to the optimizer
        if update == 0:
            self.g_h_optimizer.apply_gradients(g_h_gradients, self.g_net.trainable_variables+self.h_net.trainable_variables)
        else:
            self.d_optimizer.apply_gradients(d_gradients, self.dz_net.trainable_variables+self.dx_net.trainable_variables)
        return g_loss_adv, dx_loss, h_loss_adv, dz_loss, l2_loss_x, l2_loss_z, CE_loss_z, g_h_loss, d_loss
            

    # Feed all batches data for epoch training        
    def train_all(self): 
        batches_per_eval = 500
        ratio = 0.2
        batch_size = self.params['bs']
        weights = np.ones(self.params['nb_classes'], dtype=np.float64) / float(self.params['nb_classes'])
        last_weights = np.ones(self.params['nb_classes'], dtype=np.float64) / float(self.params['nb_classes'])
        diff_history=[]
        for batch_idx in range(self.params['nb_batches']):
            for _ in range(5):
                batch_z, batch_z_onehot = self.z_sampler.train(batch_size, weights)
                batch_x = self.x_sampler.train(batch_size)
                self.train_step(batch_z, batch_z_onehot,batch_x, update=1)
            batch_z, batch_z_onehot = self.z_sampler.train(batch_size, weights)
            batch_x = self.x_sampler.train(batch_size)            
            g_loss_adv, dx_loss, h_loss_adv, dz_loss, l2_loss_x, l2_loss_z, CE_loss_z, g_h_loss, d_loss = self.train_step(batch_z, batch_z_onehot,batch_x, update=0)
            if batch_idx % batches_per_eval == 0:
                print('Batch_idx [%d] g_loss_adv [%.4f] h_loss_adv [%.4f] CE_loss_z [%.4f]\
                    l2_loss_z [%.4f] l2_loss_x [%.4f] g_h_loss [%.4f] dz_loss [%.4f] dx_loss [%.4f] d_loss [%.4f]' %
                    (batch_idx, g_loss_adv, h_loss_adv, CE_loss_z, l2_loss_z, l2_loss_x, \
                    g_h_loss, dz_loss, dx_loss, d_loss))
                self.evaluate(batch_idx)
                ckpt_save_path = self.ckpt_manager.save()
                print ('Saving checkpoint for epoch {} at {}'.format(batch_idx,ckpt_save_path))
                tol = 0.02
                estimated_weights = self.estimate_weights(use_kmeans=False)
                weights = ratio*weights + (1-ratio)*estimated_weights
                weights = weights/np.sum(weights)
                diff_weights = np.mean(np.abs(last_weights-weights))
                diff_history.append(diff_weights)
                if np.min(weights)<tol:
                    weights = self.adjust_tiny_weights(weights,tol)
                last_weights = copy.copy(weights)
            if len(diff_history)>100 and np.mean(diff_history[-10:]) < 5e-3 and batch_idx>30000:
                print('Reach a stable cluster')
                self.evaluate(batch_idx)
                sys.exit()
                
    def adjust_tiny_weights(self,weights,tol):
        idx_less = np.where(weights<tol)[0]
        idx_greater = np.where(weights>=tol)[0]
        weights[idx_less] = np.array([np.random.uniform(2*tol,1./self.nb_classes) for item in idx_less])
        weights[idx_greater] = weights[idx_greater]*(1-np.sum(weights[idx_less]))/np.sum(weights[idx_greater])
        return weights   
                
    def evaluate(self,batch_idx):
        has_label = self.params['has_label']
        is_train = self.params['is_train']
        if has_label:
            data_x, label_true = self.x_sampler.load_all()
        else:
            data_x = self.x_sampler.load_all()
        data_z_, data_z_onehot_ = self.predict_z(data_x)
        label_infer = np.argmax(data_z_onehot_, axis=1)
        if has_label:
            nmi = normalized_mutual_info_score(label_true, label_infer)
            ari = adjusted_rand_score(label_true, label_infer)
            homo = homogeneity_score(label_true,label_infer)
            print('scDEC: NMI = {}, ARI = {}, Homogeneity = {}'.format(nmi,ari,homo))
            if is_train:
                f = open('%s/log.txt'%self.save_dir,'a+')
                f.write('NMI = {}\tARI = {}\tHomogeneity = {}\t batch_idx = {}\n'.format(nmi,ari,homo,batch_idx))
                f.close()
                np.savez('{}/data_at_{}.npz'.format(self.save_dir, batch_idx+1),data_z_,data_z_onehot_,label_true)
            else:
                np.savez('results/{}/data_pre.npz'.format(self.data),data_z_,data_z_onehot_,label_true)    
        else:
            if is_train:
                np.savez('{}/data_at_{}.npz'.format(self.save_dir, batch_idx+1),data_z_,data_z_onehot_)
            else:
                np.savez('results/{}/data_pre.npz'.format(self.data),data_z_,data_z_onehot_)
        
    #predict with x_=G(z)
    def predict_x(self, z, z_onehot, bs=256):
        assert z.shape[-1] == self.params['z_dim']
        N = z.shape[0]
        x_pred = np.zeros(shape=(N, self.params['x_dim'])) 
        for b in range(int(np.ceil(N*1.0 / bs))):
            if (b+1)*bs > N:
               ind = np.arange(b*bs, N)
            else:
               ind = np.arange(b*bs, (b+1)*bs)
            batch_z = z[ind, :]
            batch_z_onehot = z_onehot[ind, :]
            batch_x_ = self.g_net(np.concatenate([batch_z, batch_z_onehot], axis=1))
            x_pred[ind, :] = batch_x_
        return x_pred
    
    #predict with z_=H(x)
    def predict_z(self,x,bs=256):
        #return 
        assert x.shape[-1] == self.params['x_dim']
        N = x.shape[0]
        z_pred = np.zeros(shape=(N, self.params['x_dim']+self.params['nb_classes'])) 
        z_onehot = np.zeros(shape=(N, self.params['nb_classes'])) 
        for b in range(int(np.ceil(N*1.0 / bs))):
            if (b+1)*bs > N:
               ind = np.arange(b*bs, N)
            else:
               ind = np.arange(b*bs, (b+1)*bs)
            batch_x = x[ind, :]
            batch_z_,batch_z_onehot_ = self.h_net(batch_x)
            z_pred[ind, :] = batch_z_
            z_onehot[ind, :] = batch_z_onehot_
        return z_pred, z_onehot
        
        
        
        
                
            
            
            
            
            
            
            
                
                
                
                
                
                
                
                
                
                
                
            
            
        
        
