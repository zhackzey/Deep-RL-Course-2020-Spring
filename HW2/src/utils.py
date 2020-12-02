# author: Zhu Zeyu
# stuID: 1901111360
'''
This script implements utility functions
'''
import numpy as np
import tensorflow as tf
from collections import namedtuple

def ortho_init(scale=1.0):
    def _ortho_init(shape, dtype, partition_info=None):
        #lasagne ortho init for tf
        shape = tuple(shape)
        if len(shape) == 2:
            flat_shape = shape
        elif len(shape) == 4: # assumes NHWC
            flat_shape = (np.prod(shape[:-1]), shape[-1])
        else:
            raise NotImplementedError
        a = np.random.normal(0.0, 1.0, flat_shape)
        u, _, v = np.linalg.svd(a, full_matrices=False)
        q = u if u.shape == flat_shape else v # pick the one with the correct shape
        q = q.reshape(shape)
        return (scale * q[:shape[0], :shape[1]]).astype(np.float32)
    return _ortho_init

def fc(x, scope, nh, *, init_scale=1.0, init_bias=0.0):
    with tf.variable_scope(scope):
        nin = x.get_shape()[1].value
        w = tf.get_variable("w", [nin, nh], initializer=ortho_init(init_scale))
        b = tf.get_variable("b", [nh], initializer=tf.constant_initializer(init_bias))
        return tf.matmul(x, w)+b

def mlp(num_layers = 2, num_hidden = 128, activation = tf.nn.relu, layer_norm = False):
    """
    Multi-layer Perceptron. Used for Policy/Value network.
    
    param num_layers: int, number of fully connected layers
    param num_hidden: int, hidden neurons of one layer
    param activation: activation function
    param layer_norm: bool, whether perform layer normalization

    return: function which builds mlp network with given input 
    """
    def network_fn(X):
        layer = tf.layers.flatten(X)
        for i in range(num_layers):
            layer = fc(layer, 'mlp_fc{}'.format(i), nh = num_hidden, init_scale = np.sqrt(2.))
            if layer_norm:
                layer = tf.contrib.layers.layer_norm(layer, center = True, scale = True)
            layer = activation(layer)
        return layer
    
    return network_fn
    
class RunningStat(object):
    '''
    class performing state representation normalization
    '''
    def __init__(self,shape):
        self._n = 0 # sample num
        self._M = np.zeros(shape) # running mean
        self._S = np.zeros(shape) # running sum(x - mean)^2
        self.shape = shape
    
    def push(self, x):
        '''
        add one new sample x, update stats
        '''
        x = np.asarray(x)
        assert x.shape == self.shape
        self._n += 1
        if self._n == 1:
            self._M[...] = x
        else:
            oldM = self._M.copy()
            self._M[...] = oldM + (x - oldM) / self._n
            self._S[...] = self._S + (x - oldM) * (x - self._M)

    @property
    def n(self):
        return self._n
    
    @property
    def mean(self):
        return self._M

    @property
    def var(self):
        if self._n > 1:
            return self._S / (self._n -1)
        else:
            return np.square(self._M)
            #return np.zeros(self._M.shape)
    @property
    def std(self):
        return np.sqrt(self.var)

class InputNormalization:
    """
    Input X
    Output Y
    Y = (X - X_Mean) / X_Std
    """
    def __init__(self, shape, norm_mean = True, norm_std = True, clip = 5.0):
        self.norm_mean = norm_mean
        self.norm_std = norm_std
        self.clip = clip
        self.rs = RunningStat(shape)
    
    def __call__(self, x, update = True):
        if update:
            self.rs.push(x)
        if self.norm_mean:
            x = x - self.rs.mean
        if self.norm_std:
            x = x / ( self.rs.std + 1e-8 )
        if self.clip:
            x = np.clip(x, - self.clip, self.clip)
        return x

Experience = namedtuple('Experience', ('state','value','action','logproba','mask','next_state','reward'))

class Dataset(object):
    def __init__(self):
        self.dataset = []
    
    def push(self, *args):
        self.dataset.append(Experience(*args))
    
    def data(self):
        return Experience(*zip(*self.dataset))
    
    def size(self):
        return len(self.dataset)