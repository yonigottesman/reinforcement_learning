from collections import deque

import numpy as np


class StackedFrames:
    def __init__(self, stack_size, frame_shape):
        self.stacked_frames = deque([np.zeros(frame_shape, dtype=np.int)
                                     for i in range(stack_size)], maxlen=stack_size)
        self.stack_size = stack_size

    def push_get(self, frame, reset=False):
        if reset:
            self.stacked_frames = deque([frame for i in range(4)], self.stack_size)
        else:
            self.stacked_frames.append(frame)
        return np.stack(self.stacked_frames, axis=2)

