# author: Zhu Zeyu
# stuID: 1901111360
'''
This script implements CategoricalPd class
'''
import tensorflow as tf
from utils import fc, ortho_init

class CategoricalPdType(object):
    '''
    Categorical Probability Distribution Type. i.e. discrete action space
    '''
    def __init__(self, ncat):
        '''
        param ncat: int, the number of categories in the distribution
        '''
        self.ncat = ncat
    
    def param_shape(self):
        return [self.ncat]
    
    def param_placeholder(self, prepend_shape, name = None):
        return tf.placeholder(dtype = tf.float32,
                              shape = prepend_shape + self.param_shape(),
                              name = name)

    def sample_shape(self):
        '''
        in CategoricalPd, the sample is a discrete int.
        '''
        return []
    
    def sample_placeholder(self, prepend_shape, name = None):
        return tf.placeholder(dtype = self.sample_dtype(),
                              shape = prepend_shape + self.sample_shape(),
                              name = name)

    def sample_dtype(self):
        return tf.int32

    def pdfromlatent(self, latent_vector, init_scale = 1.0, init_bias = 0.0):
        '''
        construct pd from latent vector
        '''
        if latent_vector.shape[-1] == self.ncat:
            # shape matches, the latent vector is same as category number
            pdparam = latent_vector
        else:
            pdparam = fc(latent_vector,  'q', self.ncat, init_scale=init_scale, init_bias=init_bias)
            #pdparam = tf.layers.dense(latent_vector, self.ncat, activation = None, kernel_initializer = ortho_init(init_scale), bias_initializer = tf.constant_initializer(init_bias), name = 'q')
        return CategoricalPd(pdparam), pdparam

class CategoricalPd(object):
    '''
    Categorical Probability Distribution. i.e. discrete action space
    '''    
    def __init__(self,logits):
        '''
        param logits: tensor, network output for category prediction
        '''
        self.logits = logits
    
    def flatparam(self):
        '''
        here suppose logits is flatterned
        '''
        return self.logits
    
    def mode(self):
        return tf.argmax(self.logits, axis = -1)
    
    @property
    def mean(self):
        return tf.nn.softmax(self.logits)
    
    def neglogp(self, x):
        # return tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits, labels=x)
        # Note: we can't use sparse_softmax_cross_entropy_with_logits because
        #       the implementation does not allow second-order derivatives...
        if x.dtype in {tf.uint8, tf.int32, tf.int64}:
            # one-hot encoding
            x_shape_list = x.shape.as_list()
            logits_shape_list = self.logits.get_shape().as_list()[:-1]
            for xs, ls in zip(x_shape_list, logits_shape_list):
                if xs is not None and ls is not None:
                    assert xs == ls, 'shape mismatch: {} in x vs {} in logits'.format(xs, ls)

            x = tf.one_hot(x, self.logits.get_shape().as_list()[-1])
        else:
            # already encoded
            assert x.shape.as_list() == self.logits.shape.as_list()

        return tf.nn.softmax_cross_entropy_with_logits_v2(
            logits=self.logits,
            labels=x)
    def kl(self, other):
        a0 = self.logits - tf.reduce_max(self.logits, axis=-1, keepdims=True)
        a1 = other.logits - tf.reduce_max(other.logits, axis=-1, keepdims=True)
        ea0 = tf.exp(a0)
        ea1 = tf.exp(a1)
        z0 = tf.reduce_sum(ea0, axis=-1, keepdims=True)
        z1 = tf.reduce_sum(ea1, axis=-1, keepdims=True)
        p0 = ea0 / z0
        return tf.reduce_sum(p0 * (a0 - tf.log(z0) - a1 + tf.log(z1)), axis=-1)
    def entropy(self):
        a0 = self.logits - tf.reduce_max(self.logits, axis=-1, keepdims=True)
        ea0 = tf.exp(a0)
        z0 = tf.reduce_sum(ea0, axis=-1, keepdims=True)
        p0 = ea0 / z0
        return tf.reduce_sum(p0 * (tf.log(z0) - a0), axis=-1)
    def sample(self):
        '''
        sample an action from distribution
        '''
        u = tf.random_uniform(tf.shape(self.logits), dtype=self.logits.dtype)
        return tf.argmax(self.logits - tf.log(-tf.log(u)), axis=-1)