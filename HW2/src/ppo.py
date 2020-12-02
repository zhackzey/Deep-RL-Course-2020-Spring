# author: Zhu Zeyu
# stuID: 1901111360
'''
This script implements PPO computation graph
'''
import os
import re
import shutil

import numpy as np
import tensorflow as tf

from policy import ActorCritic

class ProximalPolicyOptimization():
    """
        PPO Model Class.
        Implements loss, optimizer, etc for ppo algorithm
    """
    def __init__(self, input_shape, num_actions,  entropy_coef, value_coef, model_name, model_checkpoint = None,max_grad_norm = None):
        """
            input_shape [1]:
                Shape of input states as a tuple (len,)
            num_actions (int):
                Number of discrete actions
            entropy_coef (float):
                Entropy loss coefficient
            value_coef (float):
                Value loss coefficient
            model_name (string):
                model name
            model_checkpoint (string):
                model checkpoint path
            max_grad_norm:
                Max gradient norm
        """
        tf.reset_default_graph()
        self.num_actions = num_actions

        # create the placeholders
        self.input_states = tf.placeholder(shape = (None, *input_shape),
                                           dtype =  tf.float32,
                                           name = "input_states_placeholder")

        self.taken_actions = tf.placeholder(shape = (None, ),
                                            dtype = tf.int32,
                                            name = "taken_actions_placeholder")
        self.advantages = tf.placeholder(shape = (None,),
                                        dtype = tf.float32,
                                        name = "advantages_placeholder")
        self.returns = tf.placeholder(shape = (None,),
                                      dtype = tf.float32,
                                      name = "returns_placeholder")
        
        # learning rate
        self.LR = LR = tf.placeholder(tf.float32,[])
        # Cliprange
        self.CLIPRANGE = CLIPRANGE = tf.placeholder(tf.float32, [])
        # numerical check
        self.input_states = tf.check_numerics(self.input_states, "Invalid value for self.input_states")
        self.advantages = tf.check_numerics(self.advantages, "Invalid value for self.advantages")
        self.returns = tf.check_numerics(self.returns, "Invalid value for self.returns")

        self.policy = ActorCritic(self.input_states, self.taken_actions, self.num_actions,"policy", shared_network = True, layer_norm= True)
        self.policy_old = ActorCritic(self.input_states, self.taken_actions, self.num_actions, "policy_old",shared_network = True, layer_norm = True)

        self.action_policy = self.policy.pd.sample()
        self.value_policy = self.policy.value
        self.neglogp_policy = self.policy.pd.neglogp(self.action_policy)

        self.action_policy_old = self.policy_old.pd.sample()
        self.value_policy_old = self.policy_old.value
        self.neglogp_policy_old = self.policy_old.pd.neglogp(self.action_policy_old)

        # Calculate ratio:
        # r_t(theta) = exp(log (pi(a_t |s_t, theta) - log (pi(a_t|s_t, theta_old))))
        self.prob_ratio = tf.exp( - self.policy.pd.neglogp(self.taken_actions) + self.policy_old.pd.neglogp(self.taken_actions))

        # Policy loss
        pg_loss = - self.advantages * self.prob_ratio
        pg_loss2 = - self.advantages * tf.clip_by_value(self.prob_ratio, 1.0 - CLIPRANGE, 1.0 + CLIPRANGE) 
        self.policy_loss = tf.reduce_mean(tf.maximum(pg_loss, pg_loss2))

        # value loss
        self.value_loss = tf.reduce_mean(tf.square(self.policy.value - self.returns)) * value_coef

        # entropy loss
        self.entropy_loss = tf.reduce_mean(self.policy.pd.entropy()) * entropy_coef

        # Total loss
        self.total_loss = self.policy_loss - self.entropy_loss  + self.value_loss 

        # 1.Get the model parameters
        policy_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope = "policy/")
        policy_old_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope ="policy_old/")

        # check
        assert len(policy_params) == len(policy_old_params)
        for src, dst in zip(policy_params, policy_old_params):
            assert (src.shape == dst.shape)
        
        # 2.Build optimizer
        self.optimizer = tf.train.AdamOptimizer(learning_rate=LR, epsilon= 1e-5)

        # 3.Calculate the gradients
        # Cautious: here we are optimizing on the current parameters (policy_params)
        grads_and_var = self.optimizer.compute_gradients(self.total_loss,policy_params)
        grads, var = zip(*grads_and_var)

        if max_grad_norm is not None:
            # clip the gradients
            grads, _grad_norm = tf.clip_by_global_norm(grads, max_grad_norm)
        grads_and_var = list(zip(grads,var))

        self.grads = grads
        self.var = var
        self.train_op = self.optimizer.apply_gradients(grads_and_var)
        self.update_op = tf.group([dst.assign(src) for src, dst in zip(policy_params, policy_old_params)])

        # Session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config = config)

        # Initializer
        self.sess.run(tf.global_variables_initializer())

        # Summaries
        tf.summary.scalar("policy_loss", self.policy_loss) # suppose decrease
        tf.summary.scalar("value_loss",self.value_loss) # suppose decrease
        tf.summary.scalar("entropy_loss", self.entropy_loss) # suppose increase
        tf.summary.scalar("total_loss", self.total_loss)

        tf.summary.scalar("prob_ratio",tf.reduce_mean(self.prob_ratio))
        tf.summary.scalar("returns", tf.reduce_mean(self.returns))
        tf.summary.scalar("advantages",tf.reduce_mean(self.advantages))
        tf.summary.scalar("learning_rate", self.LR)
        tf.summary.scalar("cliprange", self.CLIPRANGE)

        self.summary_op = tf.summary.merge_all()

        # load model checkpoint
        self.model_name = model_name
        self.saver = tf.train.Saver(max_to_keep = 50)

        self.model_dir = "../models/{}".format(self.model_name)
        self.log_dir = "../logs/{}".format(self.model_name)
        
        if model_checkpoint is None and os.path.isdir(self.model_dir):
            answer = input("{} exists. Continue training (C) or restart training (R)".format(self.model_dir))
            if answer.upper() == "C":
                model_checkpoint = tf.train.latest_checkpoint(self.model_dir)
            elif answer.upper() != "R":
                raise Exception("Model directory {} already exists".format(self.model_dir))
        
        if model_checkpoint:
            self.step_idx = int(re.findall(r"[/\\]step\d+", model_checkpoint)[0][len("/step"):])
            self.saver.restore(self.sess, model_checkpoint)
            print("Model checkpoint restored from {}".format(model_checkpoint))
        else:
            self.step_idx = 0
            for d in [self.model_dir, self.log_dir]:
                if os.path.isdir(d): shutil.rmtree(d)
                os.makedirs(d)      

        self.train_writer = tf.summary.FileWriter(self.log_dir, self.sess.graph)

    def save(self):
        model_checkpoint = os.path.join(self.model_dir,"step{}.ckpt".format(self.step_idx))
        self.saver.save(self.sess, model_checkpoint)
        print("Model checkpoint saved to {}".format(model_checkpoint))

    def train(self, input_states, taken_actions, returns, advantages,  cliprange, learning_rate =3e-4):
        res = self.sess.run([self.summary_op, self.train_op, self.total_loss, self.policy_loss, self.value_loss, self.entropy_loss],
                        feed_dict = {
                            self.input_states : input_states,
                            self.taken_actions : taken_actions,
                            self.returns: returns,
                            self.advantages: advantages,
                            self.CLIPRANGE: cliprange(self.step_idx) if callable(cliprange) else cliprange,
                            self.LR: learning_rate(self.step_idx) if callable(learning_rate) else learning_rate
                        })

        self.train_writer.add_summary(res[0], self.step_idx)
        self.step_idx += 1
        # return: total_loss, policy_loss, value_loss, entropy_loss
        return res[2:]
    def update_old_policy(self):
        self.sess.run(self.update_op)
        
    def predict(self, input_states, use_policy_old = False):
        '''
        if use_policy_old:
            policy = self.policy_old
        else:
            policy = self.policy
       
        action = policy.pd.sample()
        value = policy.value
        neglogp = policy.pd.neglogp(action)
        '''
        if use_policy_old:
            return self.sess.run([self.action_policy_old,self.value_policy_old,self.neglogp_policy_old],
                                 feed_dict={self.input_states : input_states})
        return self.sess.run([self.action_policy, self.value_policy, self.neglogp_policy],
                            feed_dict={
                                self.input_states: input_states,
                            })

    def write_to_summary(self, name, value, episode):
        summary = tf.Summary()
        summary.value.add(tag=name, simple_value=value)
        self.train_writer.add_summary(summary, episode)