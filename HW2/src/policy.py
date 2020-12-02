# author: Zhu Zeyu
# stuID: 1901111360
'''
This script implements Policy and Value Networks
'''
import os
import re
import shutil

import numpy as np
import tensorflow as tf

from utils import mlp,fc, ortho_init
from distribution import CategoricalPdType, CategoricalPd

class ActorCritic(object):
    """
    Actor-Critic network object offering methods for policy and value estimation.
    Actor and Critic share part of parameters.
    """
    def __init__(self, input_states, taken_actions,
                 num_actions,scope_name,  shared_network = True, layer_norm = True):
        """
            env:
                RL environment
            input_states [batch_size, obs_size]:
                Input state vectors to predict actions for
            taken_actions [batch_size, 1]:
                Actions taken by the old policy (used for training)
            num_actions (int):
                Number of discrete actions
            scope_name (string):
                scope name (i.e. policy or policy_old)
            shared_network (bool):
                Whether Actor and critic share part of network
            layer_norm(bool): 
                perform layer_norm
        """

        with tf.variable_scope(scope_name):
            # construct mlp networks
            self.policy_latent = mlp(num_layers = 2, num_hidden = 128, activation = tf.nn.relu, layer_norm = layer_norm)(input_states)
            '''
            layer = tf.layers.flatten(input_states)
            for i in range(2):
                layer = tf.layers.dense(layer, 128, activation = None, kernel_initializer = ortho_init(np.sqrt(2.0)), bias_initializer = tf.constant_initializer(0.0), name = "mlp_fc{}".format(i))
                if layer_norm:
                    layer = tf.contrib.layers.layer_norm(layer, center = True, scale = True)
                layer = tf.nn.relu(layer)
            self.policy_latent = layer
            '''
            if shared_network:
                self.value_latent = self.policy_latent
            else:
                self.value_latent = mlp(num_layers = 2, num_hidden =128, activation = tf.nn.relu, layer_norm = layer_norm)(input_states)
                '''
                v_layer = tf.layers.flatten(input_states)
                for i in range(2):
                    v_layer = tf.layers.dense(v_layer, 128, activation = None, kernel_initializer = ortho_init(np.sqrt(2.0)), bias_initializer = tf.constant_initializer(0.0),name = "mlp_fc{}".format(i))
                    if layer_norm:
                        v_layer = tf.contrib.layers.layer_norm(v_layer, center = True, scale = True)
                    v_layer = tf.nn.relu(v_layer)
                
                self.value_latent = v_layer
                '''
            # Additional Flatten Layers(may be useless)
            self.value_latent = tf.layers.flatten(self.value_latent)
            self.policy_latent = tf.layers.flatten(self.policy_latent)
        
            # ============================   Policy Branch Pi(a_t | s_t; theta)
            # create graph for sampling actions
            # latent_vector (Batch_Size, 128) --> fc -->  pdparams (Batch_Size, self.ncat) --> softmax --> logits (Batch_Size, self.ncat) (probability of each action)
            self.pdtype = CategoricalPdType(num_actions)

            self.pd, self.pi = self.pdtype.pdfromlatent(self.policy_latent, init_scale = 0.01)

            # Take an action from policy's distribution
            self.action = self.pd.sample()

            # ============================   Value Branch V(s_t; theta)
            # Note fc has no activation
            # Shape: [Batch_Size, 1]
            self.value = fc(self.value_latent,'v',1) 
            #self.value = tf.layers.dense(self.value_latent, 1, activation = None, kernel_initializer = ortho_init(np.sqrt(2.0)), bias_initializer = tf.constant_initializer(0.0),name = 'v')
            
            # Shape: [Batch_Size]
            self.value = self.value[:,0]

            # check numericals
            self.pi = tf.check_numerics(self.pi, "Invalid value for self.pi")
            self.value = tf.check_numerics(self.value, "Invalid value for self.value")

    