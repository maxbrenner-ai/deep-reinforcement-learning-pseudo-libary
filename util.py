# Random utility classes and methods
from enum import Enum
import numpy as np
import tensorflow as tf


# The colors will auto be converted to grayscale since I allow for screen_stacking (if i allowed RGB and screen stacking
# then Conv4d would actually need to be used, and thats overkill), the screen height and width can be resized as well
class StateProcessor:
    # colors can be 'gray' or 'RGB'
    # So since some state shapes just have vectors of numbers, nothing will be applied
    # Send in none for new_height and/or new_width if you dont want it changed from original shape
    def __init__(self, original_shape, new_height=84, new_width=84, screen_stack=4, colors='gray'):
        self.screen_stack = screen_stack
        if colors is 'RGB' and self.screen_stack is not None:  # Keep this like it is
            raise ValueError("If using RGB, can only have screen stack set to None, i.e. no stacking")
        if len(original_shape) is 2 or len(original_shape) > 3:
            raise ValueError("Cant process states that don't have shape of 1 or 3")

        self.original_shape = original_shape
        if len(original_shape) is 1:
            self.state_shape = original_shape
            self.process = False
            self.colors = None
        else:
            height = new_height if new_height is not None else original_shape[0]
            width = new_width if new_width is not None else original_shape[1]
            self.state_shape = (height, width, 3) if colors is 'RGB' else (height, width, screen_stack)
            self.colors = colors
            self.process = True
            self.create_graph()

    # Using TF for some operations so need to make a graph to run a session
    def create_graph(self):
        self.input_state = tf.placeholder(tf.float32, shape=self.original_shape, name='state_proc_input_state')
        self.proc_state = self.input_state
        # Convert to grayscale
        if self.colors is 'gray':
            self.proc_state = tf.image.rgb_to_grayscale(self.input_state)

        # Normalize [0, 1]
        self.proc_state = tf.to_float(self.proc_state) / 255.

        # Resize
        self.proc_state = tf.image.resize_images(self.proc_state, [self.state_shape[0], self.state_shape[1]], method=tf.image.ResizeMethod.BILINEAR)
        self.proc_state = tf.squeeze(self.proc_state)

    # Used when a new state (or initial state) is spit out by the env.
    def process_state(self, next_state, sess, current_state):
        if not self.process:
            return next_state
        # Recolor, resize ect.
        return_state = sess.run(self.proc_state, {self.input_state: next_state})
        # Stack
        if self.screen_stack is not None:
            # If current_state is none that means we were sent a reset state
            if current_state is None:
                return_state = np.stack([return_state] * self.screen_stack, axis=2)
            else:
                # So we get rid of the furthest ago state we added with current_state[:,:,1:], then add a dim to the
                # next_state with np.expand_dims(next_state, 2), then we add that to the end of the current_state with
                # the append
                return_state = np.append(current_state[:,:,1:], np.expand_dims(return_state, axis=2), axis=2)
        return return_state

    def summary(self):
        text = ""
        text += "Shape used (H x W x Channels): {}\n".format(self.state_shape)
        text += "Screen stack: {}\n".format(self.screen_stack)
        text += "Colors: {}\n".format(self.colors)
        return text

class RunType(Enum):
    RAND_FILL = 1
    TRAIN = 2
    TEST = 3


class AgentType(Enum):
    DQN = 'Deep Q-Network (DQN)'


'''
This sum tree implementation is from:
https://github.com/jaara/AI-blog/blob/master/SumTree.py
'''

class SumTree:
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros( 2*capacity - 1 )
        self.data = np.zeros( capacity, dtype=object )
        self.write = 0
        self.is_full = False

    def _propagate(self, idx, change):
        parent = (idx - 1) // 2

        self.tree[parent] += change

        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1

        if left >= len(self.tree):
            return idx

        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s-self.tree[left])

    def total(self):
        return self.tree[0]

    def add(self, p, data):
        idx = self.write + self.capacity - 1

        self.data[self.write] = data
        self.update(idx, p)

        self.write += 1
        if self.write >= self.capacity:
            self.write = 0
            self.is_full = True

    def update(self, idx, p):
        change = p - self.tree[idx]

        self.tree[idx] = p
        self._propagate(idx, change)

    def get(self, s):
        idx = self._retrieve(0, s)
        dataIdx = idx - self.capacity + 1
        return (idx, self.tree[idx], self.data[dataIdx])