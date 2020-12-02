# author: Zhu Zeyu
# stuID: 1901111360
'''
Implementation of Experience Replay module for DQN
'''
import sys
import os
import numpy as np

class ExperienceReplay:
    """
    Save the agent's experience in windowed cache. Each step is saved only once.
    
    Note: for academy_empty_goal and simple115 representation, i do not use stacked version here.
    If using images as input, may consider stack e.g 4 frames alternatively.
    """
    def __init__(self,
            num_frame_stack = 1,
            capacity = int(1e5),
            obs_size = (115,)):
            self.num_frame_stack = num_frame_stack
            self.capacity = capacity
            self.obs_size = obs_size
            self.counter = 0
            self.frame_window = None
            self.init_caches()
            self.expecting_new_episode = True
    
    def init_caches(self):
        self.rewards = np.zeros(self.capacity, dtype = np.float32)
        self.prev_states = -np.ones((self.capacity, self.num_frame_stack), dtype = np.int32)
        self.next_states = -np.ones((self.capacity, self.num_frame_stack), dtype = np.int32)
        self.is_done = - np.ones(self.capacity, dtype = np.int32)
        self.actions = -np.ones(self.capacity, dtype = np.int32)

        self.max_frame_num  = self.capacity + 2 * self.num_frame_stack + 1
        self.frames = -np.ones((self.max_frame_num,) + self.obs_size, dtype = np.float32)

    def new_episode(self, frame):
        """
        Set up for a new episode
        """        
        assert self.expecting_new_episode, "previous episode didn't finish"
        frame_idx = self.counter % self.max_frame_num
        self.frame_window = np.repeat(frame_idx, self.num_frame_stack)
        self.frames[frame_idx] = frame
        self.expecting_new_episode = False

    def current_state(self):
        """
        Return current state in experience replay
        """
        assert self.frame_window is not None, "frame windows is None!"
        return self.frames[self.frame_window]

    def add_experience(self, frame, action, done, reward):
        """
        Add a new experience(transition (s,a,r,s'))
        add action , done , reward into experience and update indexs
        """
        assert self.frame_window is not None, "start episodes first"
        self.counter += 1
        frame_idx = self.counter % self.max_frame_num
        exp_idx = (self.counter - 1) % self.capacity

        self.prev_states[exp_idx] = self.frame_window
        self.frame_window = np.append(self.frame_window[1:], frame_idx)
        self.next_states[exp_idx] = self.frame_window 
        self.actions[exp_idx] = action
        self.is_done[exp_idx] = done
        self.frames[frame_idx] = frame
        self.rewards[exp_idx] = reward
        if done:
            # this episode ends, expecting a new one
            self.expecting_new_episode = True


    def sample_mini_batch(self, num):
        """
        May be the most useful function in this module :)
        sample the mini-batch to feed into the Q network and compute gradients
        """
        count = min(self.capacity, self.counter)
        batchidx = np.random.randint(count, size = num)
        
        prev_frames = self.frames[self.prev_states[batchidx]]
        next_frames = self.frames[self.next_states[batchidx]]

        return {
            "reward" : self.rewards[batchidx],
            "prev_state": prev_frames,
            "next_state": next_frames,
            "actions": self.actions[batchidx],
            "done_mask": self.is_done[batchidx]
        }
    
    def print_replay_buffer(self,size):
        assert size < self.capacity and size < self.max_frame_num
        print("counter:",self.counter)
        print("frames:",self.frames[:size])
        print("frame_window/current stacked frames:",self.frame_window)
        print("next_states:",self.next_states[:size])
        print("prev_states:",self.prev_states[:size])
        print("is_done:",self.is_done[:size])
        print("expecting_new_episode:",self.expecting_new_episode)

"================================================================================================"
"Do some test below"
"================================================================================================"
sys.path.append(os.getcwd())
from unittest import TestCase


class TestExperienceReplay(TestCase):
    
    def test1_for_image(self):
        """
        test for using stacked image
        """
        num_frame_stack = 3
        size = 10
        # here suppose using (4,4) image
        obs_size = (4,4)
        rb = ExperienceReplay(num_frame_stack = num_frame_stack, capacity = size, obs_size = obs_size)

        with self.assertRaises(AssertionError):
            rb.current_state()
        
        with self.assertRaises(AssertionError):
            rb.add_experience(None,None,None,None)
        
        frames = np.random.rand(4,4,4).astype("float32")
        #frames = np.random.rand(4,115).astype("float32")

        # add the beginning frame
        rb.new_episode(frames[0])
        rb.print_replay_buffer(4)
        assert (rb.current_state() == frames[0]).all()
        assert (rb.current_state().shape == (num_frame_stack,) + obs_size)

        # Now add next frame
        # action is action taken before this frame
        # reward is the reward obtained for this action
        # done is a flag if ends
        rb.add_experience(frames[1], 1, False, 2.0)
        rb.print_replay_buffer(4)

        #assert (rb.current_state() == frames[1]).all()
        assert (rb.current_state()[:2] == frames[0]).all()
        assert (rb.current_state()[2] == frames[1]).all()
        assert (rb.current_state().shape == (num_frame_stack,) + obs_size) 

        # add one more experience and set episode as finished
        rb.add_experience(frames[2], 3, True, 4.0)
        rb.print_replay_buffer(4)

        assert (rb.current_state() == frames[:3]).all()
        assert (rb.current_state().shape == (num_frame_stack,) + obs_size) 

        assert np.all(rb.next_states[:3] == np.array([[0, 0, 1], [0, 1, 2], [-1, -1, -1]]))
        assert np.all(rb.prev_states[:3] == np.array([[0, 0, 0], [0, 0, 1], [-1, -1, -1]]))

        rb.new_episode(frames[3])
        rb.print_replay_buffer(4)

        assert (rb.current_state() == frames[3]).all()
        assert (rb.current_state().shape == (num_frame_stack,) + obs_size) 

        batch = rb.sample_mini_batch(20)

        assert np.all(np.in1d(batch['reward'], [2., 4.]))
        assert np.all(np.in1d(batch['actions'], [1., 3.]))

        dm = ~batch["done_mask"].astype(bool)
        assert np.all(batch["next_state"][dm] == np.array(frames[[0, 0, 1]]))

        assert np.all(batch["next_state"][~dm] == np.array(frames[[0, 1, 3]]))
        assert np.all((batch["prev_state"] == frames[0]) | (batch["prev_state"] == frames[1]))


    def test2_for_vector(self):
        """
        test for using vector input
        """
        num_frame_stack = 1
        size = 10
        obs_size = (5,)

        rb = ExperienceReplay(num_frame_stack = num_frame_stack, capacity = size, obs_size = obs_size)

        with self.assertRaises(AssertionError):
            rb.current_state()
        
        with self.assertRaises(AssertionError):
            rb.add_experience(None,None,None,None)
        
        shape = (4,) + obs_size
        frames = np.random.rand(*shape).astype("float32")
        #frames = np.random.rand(4,115).astype("float32")

        # add the beginning frame
        rb.new_episode(frames[0])
        rb.print_replay_buffer(4)
        assert (rb.current_state() == frames[0]).all()
        assert (rb.current_state().shape == (num_frame_stack,) + obs_size)

        # Now add next frame
        # action is action taken before this frame
        # reward is the reward obtained for this action
        # done is a flag if ends
        rb.add_experience(frames[1], 1, False, 2.0)
        rb.print_replay_buffer(4)

        #assert (rb.current_state() == frames[1]).all()
        assert (rb.current_state() == frames[1]).all()
        assert (rb.current_state().shape == (num_frame_stack,) + obs_size) 

        # add one more experience and set episode as finished
        rb.add_experience(frames[2], 3, True, 4.0)
        rb.print_replay_buffer(4)

        assert (rb.current_state() == frames[2]).all()
        assert (rb.current_state().shape == (num_frame_stack,) + obs_size) 

        assert np.all(rb.next_states[:3] == np.array([[1], [2], [-1]]))
        assert np.all(rb.prev_states[:3] == np.array([[0], [1], [-1]]))

        rb.new_episode(frames[3])
        rb.print_replay_buffer(4)

        assert (rb.current_state() == frames[3]).all()
        assert (rb.current_state().shape == (num_frame_stack,) + obs_size) 

        batch = rb.sample_mini_batch(20)

        assert np.all(np.in1d(batch['reward'], [2., 4.]))
        assert np.all(np.in1d(batch['actions'], [1., 3.]))

        dm = ~batch["done_mask"].astype(bool)
        assert np.all(batch["next_state"][dm] == np.array(frames[1]))

        assert np.all(batch["next_state"][~dm] == np.array(frames[3]))
        assert np.all((batch["prev_state"] == frames[0]) | (batch["prev_state"] == frames[1]))

        
if __name__ == "__main__": 
    t = TestExperienceReplay()
    print("test1_for_image======================================")
    t.test1_for_image()
    print("test2_for_vector=====================================")
    t.test2_for_vector()

