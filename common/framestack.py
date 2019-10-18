from collections import deque

import numpy as np
import torch


class FrameStack:
    def __init__(self, stack_size, frame_shape):
        self.stacked_frames = deque([np.zeros(frame_shape, dtype=np.int)]*stack_size, maxlen=stack_size)
        self.stack_size = stack_size

    def push_get(self, frame, reset=False):
        if reset:
            self.stacked_frames = deque([frame]*self.stack_size, self.stack_size)
        else:
            self.stacked_frames.append(frame)
        return torch.tensor(self.stacked_frames).unsqueeze(0)


