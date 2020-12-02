# author: Zhu Zeyu
# stuID: 1901111360
'''
Implementation of DQN agent for football game
Note that: in future may use stacked images as input, but for now
just use simple115(vector) as input
'''
import os
import sys
import numpy as np

import tensorflow.contrib.slim as slim
import tensorflow as tf
import itertools as it
from experience_replay import ExperienceReplay

class DQN:
    """
    DQN implementation. Note that only supports an environment that is gym-like.(i.e. reset, step, ..)
    """
    def __init__(self,
            env,
            obs_size = (115,),
            num_frame_stack = 1,
            batch_size = 32,
            mdp_gamma = 0.95,
            initial_epsilon = 1.0,
            min_epsilon = 0.1,
            epsilon_decay_steps = int(1e6),
            replay_capacity = int(1e5),
            min_replay_size = int(1e3),
            train_freq = 4,
            network_update_freq = 5000,
            regularization = 1e-6,
            optimizer_params = None,
            render = False):

            """
            Initialization function
            
            param env:                object. a gym-like environment which our RL agent interacts with
            parma obs_size:           list. the shape of the observation, i.e. (115,) for vector observation or (32,32) for image observation
            parma num_frame_stack:    int. number of stacked frames for network input
            param batch_size:         int. batch size
            param mdp_gamma:          float. MDP discount factor
            param initial_epsilon:    float. epsilon parameter of epsilon-greedy policy
            param min_epsilon:        float. minimum epsilon parameter of epsilon-greedy policy
            param epsilon_decay_steps: int. how many steps to decay epsilon 
            param replay_capacity:    int. replay buffer size
            param min_replay_size:    int. minimum replay buffer size
            param train_freq:         int. training frequency
            param network_update_freq: int. network update frequency
            param regularization:     float. regularization coefficient
            param optimizer_params:   dict. optimizer specilized parameters. i.e. learning rate, momentum
            param render:             bool. is render mode on?
            """
            
            # experience replay buffer for training
            self.exp_buffer = ExperienceReplay(
                num_frame_stack,
                capacity=replay_capacity,
                obs_size = obs_size
            )

            # experience replay buffer for playing/testing
            self.play_buffer = ExperienceReplay(
                num_frame_stack,
                capacity=num_frame_stack * 10,
                obs_size = obs_size
            )

            self.env = env
            self.obs_size = obs_size
            self.num_frame_stack = num_frame_stack
            self.batch_size = batch_size
            self.mdp_gamma = mdp_gamma
            self.initial_epsilon = initial_epsilon
            self.min_epsilon = min_epsilon
            self.epsilon_decay_steps = epsilon_decay_steps
            self.replay_capacity = replay_capacity
            self.min_replay_size = min_replay_size
            self.train_freq = train_freq
            self.network_update_freq = network_update_freq
            self.regularization = regularization
            self.render = render

            self.dim_actions = env.action_space.n
            self.dim_state = (num_frame_stack,) + self.obs_size

            if optimizer_params:
                self.optimizer_params = optimizer_params
            else:
                self.optimizer_params = dict(learning_rate = 0.0001, epsilon = 1e-7)

            self.is_training = True
            # epsilon used for playing
            # if 0, means that we just use the Q-network's optimal action without any exploration
            self.playing_epsilon = 0.0
            
            self.session = None
            
            self.global_counter = 0
            self.episode_counter = 0
            self.loss_history = []

    def get_variables(self,scope):
        """
        Get variables according to scope name
        """
        vars_list = []
        for var in tf.global_variables():
            if "%s/" % scope in var.name and "Adam" not in var.name:
                vars_list.append(var)
        return sorted(vars_list, key=lambda x: x.name)
    
    def get_epsilon(self):
        """
        Get current epsilon value.
        Note: with the training process, epsilon is decaying
        """
        if self.is_training == False:
            return self.playing_epsilon
        elif self.global_counter >= self.epsilon_decay_steps:
            return self.min_epsilon
        else:
            # for simplicity, just use linear decay
            return self.min_epsilon + (self.initial_epsilon - self.min_epsilon) * (1.0 - self.global_counter / float(self.epsilon_decay_steps))
            

    def network(self, input, trainable, use_image = False):
        """
        Implementation of Q(s,a) network
        
        param input:  tensor. [Batch_Size, N_State] or [Batch_Size, Num_stack_frame, H, W]

        """

        regularizer = None
        if trainable:
            regularizer = slim.l2_regularizer(self.regularization)
        
        if not use_image:
            # here use vanilla 4-layer perceptron
            # 1st layer
            net = slim.fully_connected(input, 512, activation_fn = tf.nn.relu, weights_regularizer = regularizer, trainable = trainable)
            # 2nd layer
            net = slim.fully_connected(net, 1024, activation_fn = tf.nn.relu, weights_regularizer = regularizer, trainable = trainable)
            # 3rd layer
            net = slim.fully_connected(net,512, activation_fn = tf.nn.relu, weights_regularizer = regularizer, trainable = trainable)
            # 4th layer
            #net = slim.fully_connected(net, 256, activation_fn = tf.nn.relu, weights_regularizer = regularizer, trainable = trainable)

            # output layer
            q_state_action_values = slim.fully_connected(net, self.dim_actions, activation_fn = None, weights_regularizer = regularizer, trainable = trainable)

        else:
            
            x = tf.transpose(input, [0,2,3,1])

            net = slim.conv2d(x, 8, (7,7),  stride = 3, data_format = "NHWC", activation_fn = tf.nn.relu, weights_regularizer = regularizer, trainable = trainable)
            net = slim.max_pool2d(net, 2, 2)
            net = slim.conv2d(net, 16, (3,3), stride = 1, data_format = "NHWC", activation_fn = tf.nn.relu, weights_regularizer = regularizer, trainable = trainable)
            net = slim.max_pool2d(net, 2, 2)
            net = slim.flatten(net)
            net = slim.fully_connected(net, 256, activation_fn = tf.nn.relu, weights_regularizer = regularizer, trainable = trainable)
            q_state_action_values = slim.fully_connected(net, self.dim_actions, activation_fn = None, weights_regularizer = regularizer, trainable = trainable)
        
        return q_state_action_values

    def sample_random_action(self):
        """
        Randomly sample an action for rollout
        """
        return np.random.choice(self.dim_actions)
    
    
    
    def setup_graph(self, use_image = False, if_soft = True):
        """
        Set up tensorflow computing graph
        """

        # define a bunch of placeholders
        if use_image:
            input_next_state_shape = (self.batch_size, self.num_frame_stack) + self.obs_size
            input_prev_state_shape = (None, self.num_frame_stack) + self.obs_size
        else:
            input_next_state_shape = (self.batch_size, self.obs_size[0])
            input_prev_state_shape = (None, self.obs_size[0])

        self.input_prev_state = tf.placeholder(tf.float32, input_prev_state_shape, name = "input_prev_state")
        self.input_next_state = tf.placeholder(tf.float32, input_next_state_shape, name = "input_next_state")
        self.input_actions = tf.placeholder(tf.int32, self.batch_size, name = "input_actions")
        self.input_reward = tf.placeholder(tf.float32, self.batch_size, name = "input_reward")
        self.is_done = tf.placeholder(tf.int32, self.batch_size, name = "is_done")

        self.optimizer = tf.train.AdamOptimizer(**(self.optimizer_params))
        """
        Q-learning:
        1. take action a_t according to epsilon-greedy policy
        2. store transition (s_t, a_t, r_t+1, s_t+1) in replay buffer D
        3. sample random mini-batch of transitions (s,a,r,s') from D
        3. compute Q-learning targets w.r.t. old, fixed parameters w-
        4. optimise MSE between Q-network and Q-learning targets

        L(w) = E{s,a,r,s' ~ D} [(r + \gamma \max_a'  Q(s',a',w-) - Q(s,a,w))^2]

        5. use variant of stochastic gradient descent
        """
        # Note: the following 2 networks need to have the same structure
        # fixed, old parameters Q-network for Q-target estimation
        with tf.variable_scope("target_q"):
            q_target = self.network(self.input_next_state, trainable=False, use_image = use_image)
        
        # trainable, new parameters Q-network for Q-learning
        with tf.variable_scope("update_q"):
            q_estimate = self.network(self.input_prev_state, trainable=True, use_image = use_image)
        # optimal action recovered by newest Q-network
        self.optimal_action = tf.argmax(q_estimate, axis = 1)
        
        not_done = tf.cast(tf.logical_not(tf.cast(self.is_done, "bool")), tf.float32)
        q_target_value = self.input_reward + not_done * self.mdp_gamma * tf.reduce_max(q_target, -1)

        # choose chosen self.input_actions from q_estimate to get values
        # first get indexes
        idx = tf.stack([tf.range(0, self.batch_size), self.input_actions], axis = 1)
        q_estimate_value = tf.gather_nd(q_estimate, idx)

        # MSE loss
        mse_loss = tf.nn.l2_loss(q_estimate_value - q_target_value) / self.batch_size
        # Regularization loss
        regularization_loss = tf.add_n(tf.losses.get_regularization_losses())

        self.loss = mse_loss + regularization_loss
        self.train_op = self.optimizer.minimize(self.loss)

        update_params = self.get_variables("update_q")
        target_params = self.get_variables("target_q")

        assert (len(update_params) == len(target_params))
        # weights copy op
        if if_soft:
            self.assign_op = [tf.assign(tp,0.001 * up + 0.999 * tp) for tp, up in zip(target_params, update_params)]
        else:
            self.assign_op = [tf.assign(tp,up) for tp, up in zip(target_params, update_params)]

    def train(self):
        """
        train step
        """
        # sample one mini-batch to compute mse
        batch = self.exp_buffer.sample_mini_batch(self.batch_size)
        if self.num_frame_stack > 1:
            # suppose use image observation
            feed_dict = {
                self.input_prev_state : batch["prev_state"],
                self.input_next_state : batch["next_state"],
                self.input_actions: batch["actions"],
                self.is_done: batch["done_mask"],
                self.input_reward: batch["reward"]
            }
        else:
            # reduce the axis 1
            feed_dict = {
                self.input_prev_state : batch["prev_state"][:,0,:],
                self.input_next_state : batch["next_state"][:,0,:],
                self.input_actions: batch["actions"],
                self.is_done: batch["done_mask"],
                self.input_reward: batch["reward"]
            }

        _, loss = self.session.run([self.train_op, self.loss], feed_dict=feed_dict)
        self.loss_history.append(loss)

        return loss

    def update_target_network(self):
        """
        Update target network
        """
        # no need for feed dicts
        self.session.run(self.assign_op)

    def play_episode(self):
        if self.is_training:
            rb = self.exp_buffer
        else:
            rb = self.play_buffer
        
        # total reward
        sum_reward = 0
        # total loss
        sum_loss = 0
        # steps
        steps_in_episode = 0

        first_obs = self.env.reset()
        rb.new_episode(first_obs)

        while True:
            if np.random.rand() > self.get_epsilon():
                if self.num_frame_stack > 1:
                    action = self.session.run(self.optimal_action, {self.input_prev_state: rb.current_state()[np.newaxis,:]})[0]
                else:
                    action = self.session.run(self.optimal_action, {self.input_prev_state: rb.current_state()})[0]
            else:
                action = self.sample_random_action()
             
            obs, reward, done, info = self.env.step(action)
            if self.render:
                self.env.render()
            else:
                pass
            
            sum_reward += reward
            steps_in_episode += 1

            # add one experience into buffer
            rb.add_experience(obs, action, done, reward)

            if self.is_training:
                self.global_counter += 1
                if self.global_counter % self.network_update_freq == 0:
                    self.update_target_network()
                if self.exp_buffer.counter >= self.min_replay_size and self.global_counter % self.train_freq == 0:
                    sum_loss += self.train()
            if done:
                if self.is_training:
                    self.episode_counter += 1
                
                return sum_reward, steps_in_episode, sum_loss / float(steps_in_episode)